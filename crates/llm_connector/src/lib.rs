// crates/llm_connector/src/lib.rs

use async_trait::async_trait;
use klarnet_core::{KlarnetError, KlarnetResult};
use parking_lot::RwLock;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, error, info, warn};

pub mod openrouter;
pub mod deepseek;
pub mod cache;
pub mod prompt_builder;

use cache::LlmCache;
use prompt_builder::PromptBuilder;

/// LLM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: LlmProvider,
    pub model: String,
    pub api_key_env: String,
    pub base_url: Option<String>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub timeout_s: u64,
    pub retry_attempts: u32,
    pub cache_enabled: bool,
    pub cache_ttl_s: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LlmProvider {
    OpenRouter,
    DeepSeek,
    OpenAI,
    Custom(String),
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: LlmProvider::OpenRouter,
            model: "deepseek/deepseek-chat".to_string(),
            api_key_env: "OPENROUTER_API_KEY".to_string(),
            base_url: None,
            max_tokens: 500,
            temperature: 0.3,
            top_p: 0.95,
            timeout_s: 10,
            retry_attempts: 3,
            cache_enabled: true,
            cache_ttl_s: 3600,
        }
    }
}

/// LLM provider trait
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(&self, request: CompletionRequest) -> KlarnetResult<CompletionResponse>;
    fn name(&self) -> &str;
}

/// Completion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub messages: Vec<Message>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop: Option<Vec<String>>,
    pub functions: Option<Vec<Function>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Function,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// Completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub content: String,
    pub function_call: Option<FunctionCall>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// LLM connector
pub struct LlmConnector {
    config: LlmConfig,
    provider: Box<dyn LlmProvider>,
    cache: Option<LlmCache>,
    prompt_builder: PromptBuilder,
    metrics: Arc<RwLock<LlmMetrics>>,
}

#[derive(Debug, Default)]
struct LlmMetrics {
    total_requests: u64,
    successful_requests: u64,
    failed_requests: u64,
    cache_hits: u64,
    total_tokens_used: usize,
    average_response_time_ms: f64,
}

impl LlmConnector {
    pub async fn new(config: LlmConfig) -> KlarnetResult<Self> {
        let provider: Box<dyn LlmProvider> = match &config.provider {
            LlmProvider::OpenRouter => {
                Box::new(openrouter::OpenRouterProvider::new(config.clone()).await?)
            }
            LlmProvider::DeepSeek => {
                Box::new(deepseek::DeepSeekProvider::new(config.clone()).await?)
            }
            _ => {
                return Err(KlarnetError::Nlu("Unsupported LLM provider".to_string()));
            }
        };

        let cache = if config.cache_enabled {
            Some(LlmCache::new(config.cache_ttl_s))
        } else {
            None
        };

        let prompt_builder = PromptBuilder::new();
        let metrics = Arc::new(RwLock::new(LlmMetrics::default()));

        Ok(Self {
            config,
            provider,
            cache,
            prompt_builder,
            metrics,
        })
    }

    pub async fn complete(&self, request: CompletionRequest) -> KlarnetResult<CompletionResponse> {
        let start = std::time::Instant::now();

        // Check cache
        let cache_key = self.generate_cache_key(&request);
        if let Some(cache) = &self.cache {
            if let Some(response) = cache.get(&cache_key) {
                self.metrics.write().cache_hits += 1;
                debug!("LLM cache hit");
                return Ok(response);
            }
        }

        // Make request with retries
        let mut attempts = 0;
        let mut last_error = None;

        while attempts < self.config.retry_attempts {
            match self.provider.complete(request.clone()).await {
                Ok(response) => {
                    // Update metrics
                    let duration = start.elapsed();
                    let mut metrics = self.metrics.write();
                    metrics.total_requests += 1;
                    metrics.successful_requests += 1;
                    metrics.total_tokens_used += response.usage.total_tokens;
                    metrics.average_response_time_ms =
                        (metrics.average_response_time_ms * (metrics.successful_requests - 1) as f64
                            + duration.as_millis() as f64) / metrics.successful_requests as f64;

                    // Cache response
                    if let Some(cache) = &self.cache {
                        cache.set(cache_key, response.clone());
                    }

                    return Ok(response);
                }
                Err(e) => {
                    attempts += 1;
                    last_error = Some(e);

                    if attempts < self.config.retry_attempts {
                        let delay = Duration::from_millis(100 * 2u64.pow(attempts));
                        warn!("LLM request failed, retrying in {:?}", delay);
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        self.metrics.write().failed_requests += 1;
        Err(last_error.unwrap())
    }

    pub async fn process_command(&self, text: &str) -> KlarnetResult<CommandInterpretation> {
        let prompt = self.prompt_builder.build_command_prompt(text);

        let request = CompletionRequest {
            messages: vec![
                Message {
                    role: Role::System,
                    content: prompt,
                },
                Message {
                    role: Role::User,
                    content: text.to_string(),
                },
            ],
            max_tokens: Some(self.config.max_tokens),
            temperature: Some(self.config.temperature),
            top_p: Some(self.config.top_p),
            stop: None,
            functions: Some(self.get_available_functions()),
        };

        let response = self.complete(request).await?;

        self.parse_command_response(response)
    }

    fn generate_cache_key(&self, request: &CompletionRequest) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        format!("{:?}", request).hash(&mut hasher);
        format!("llm:{}", hasher.finish())
    }

    fn get_available_functions(&self) -> Vec<Function> {
        vec![
            Function {
                name: "execute_command".to_string(),
                description: "Execute a system or smart home command".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "The action to execute"
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Parameters for the action"
                        }
                    },
                    "required": ["action"]
                }),
            },
            Function {
                name: "answer_question".to_string(),
                description: "Provide an answer to a question".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The answer to the question"
                        }
                    },
                    "required": ["answer"]
                }),
            },
        ]
    }

    fn parse_command_response(&self, response: CompletionResponse) -> KlarnetResult<CommandInterpretation> {
        if let Some(function_call) = response.function_call {
            let args: Value = serde_json::from_str(&function_call.arguments)
                .unwrap_or_else(|_| json!({}));

            Ok(CommandInterpretation {
                intent: function_call.name,
                parameters: args,
                response_text: response.content,
                confidence: 0.9,
            })
        } else {
            // Try to parse JSON from content
            if let Ok(json) = serde_json::from_str::<Value>(&response.content) {
                Ok(CommandInterpretation {
                    intent: json["intent"].as_str().unwrap_or("unknown").to_string(),
                    parameters: json["parameters"].clone().unwrap_or(json!({})),
                    response_text: json["response"].as_str().unwrap_or(&response.content).to_string(),
                    confidence: json["confidence"].as_f64().unwrap_or(0.5) as f32,
                })
            } else {
                Ok(CommandInterpretation {
                    intent: "chat".to_string(),
                    parameters: json!({}),
                    response_text: response.content,
                    confidence: 0.5,
                })
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandInterpretation {
    pub intent: String,
    pub parameters: Value,
    pub response_text: String,
    pub confidence: f32,
}