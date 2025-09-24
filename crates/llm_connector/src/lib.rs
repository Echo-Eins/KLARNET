// crates/llm_connector/src/lib.rs
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use klarnet_core::{KlarnetError, KlarnetResult};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::warn;

pub mod cache;
pub mod deepseek;
pub mod openrouter;
pub mod prompt_builder;

use cache::LlmCache;
use deepseek::DeepSeekProvider;
use openrouter::OpenRouterProvider;
use prompt_builder::PromptBuilder;

/// LLM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: LlmProviderKind,
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
pub enum LlmProviderKind {
    OpenRouter,
    DeepSeek,
    OpenAI,
    Custom(String),
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: LlmProviderKind::OpenRouter,
            model: "x-ai/grok-4-fast:free".to_string(),
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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

#[derive(Debug, Clone, Default)]
pub struct LlmMetricsSnapshot {
    pub provider: String,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub cache_hits: u64,
    pub total_tokens_used: usize,
    pub average_response_time_ms: f64,
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
            LlmProviderKind::OpenRouter => Box::new(OpenRouterProvider::new(config.clone()).await?),
            LlmProviderKind::DeepSeek => Box::new(DeepSeekProvider::new(config.clone()).await?),
            LlmProviderKind::OpenAI | LlmProviderKind::Custom(_) => {
                return Err(KlarnetError::Nlu("Unsupported LLM provider".to_string()))
            }
        };

        let cache = if config.cache_enabled {
            Some(LlmCache::new(config.cache_ttl_s))
        } else {
            None
        };

        Ok(Self {
            config,
            provider,
            cache,
            prompt_builder: PromptBuilder::new(),
            metrics: Arc::new(RwLock::new(LlmMetrics::default())),
        })
    }
    pub fn config(&self) -> &LlmConfig {
        &self.config
    }

    pub fn metrics_snapshot(&self) -> LlmMetricsSnapshot {
        let metrics = self.metrics.read();
        LlmMetricsSnapshot {
            provider: self.provider.name().to_string(),
            total_requests: metrics.total_requests,
            successful_requests: metrics.successful_requests,
            failed_requests: metrics.failed_requests,
            cache_hits: metrics.cache_hits,
            total_tokens_used: metrics.total_tokens_used,
            average_response_time_ms: metrics.average_response_time_ms,
        }
    }

    pub fn prompt_builder(&self) -> &PromptBuilder {
        &self.prompt_builder
    }

    pub async fn complete(&self, request: CompletionRequest) -> KlarnetResult<CompletionResponse> {
        let start = Instant::now();

        // Check cache
        let cache_key = self.generate_cache_key(&request);
        if let Some(cache) = &self.cache {
            if let Some(response) = cache.get(&cache_key) {
                let mut metrics = self.metrics.write();
                metrics.total_requests += 1;
                metrics.successful_requests += 1;
                metrics.cache_hits += 1;
                return Ok(response);
            }
        }

        {
            let mut metrics = self.metrics.write();
            metrics.total_requests += 1;
        }

        let mut attempts = 0;
        let mut last_error = None;

        while attempts < self.config.retry_attempts {
            attempts += 1;
            match self.provider.complete(request.clone()).await {
                Ok(response) => {
                    // Update metrics
                    let duration = start.elapsed();
                    let mut metrics = self.metrics.write();
                    metrics.successful_requests += 1;
                    metrics.total_tokens_used += response.usage.total_tokens;
                    let successes = metrics.successful_requests.max(1);
                    metrics.average_response_time_ms = if successes == 1 {
                        duration.as_secs_f64() * 1000.0
                    } else {
                        ((metrics.average_response_time_ms * (successes - 1) as f64)
                            + duration.as_secs_f64() * 1000.0)
                            / successes as f64
                    };

                    // Cache response
                    if let Some(cache) = &self.cache {
                        cache.set(cache_key, response.clone());
                    }

                    return Ok(response);
                }
                Err(err) => {
                    last_error = Some(err);

                    if attempts < self.config.retry_attempts {
                        let delay = Duration::from_millis(100 * 2u64.pow(attempts as u32));
                        warn!("LLM request failed, retrying in {:?}", delay);
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        let mut metrics = self.metrics.write();
        metrics.failed_requests += 1;
        Err(last_error.unwrap_or_else(|| {
            KlarnetError::Nlu("LLM request failed without specific error".to_string())
        }))
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
                            },
                        "route": {
                            "type": "string",
                            "description": "Optional routing target"
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0
                        }
                    },
                    "required": ["action"],
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
                            "description": "The answer to the user question"
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0
                        }
                    },
                    "required": ["answer"],
                }),
            },
        ]
    }

    fn parse_command_response(
        &self,
        response: CompletionResponse,
    ) -> KlarnetResult<CommandInterpretation> {
        if let Some(function_call) = response.function_call.clone() {
            let args: Value =
                serde_json::from_str(&function_call.arguments).unwrap_or_else(|_| json!({}));

            let function_name = function_call.name.clone();

            let (action, route) = match function_name.as_str() {
                "execute_command" => (
                    args.get("action")
                        .and_then(Value::as_str)
                        .map(|s| s.to_string()),
                    args.get("route")
                        .and_then(Value::as_str)
                        .map(|s| s.to_string()),
                ),
                _ => (None, None),
            };

            let confidence = args
                .get("confidence")
                .and_then(Value::as_f64)
                .unwrap_or(0.9) as f32;

            let parameters = args.get("parameters").cloned().unwrap_or_else(|| json!({}));

            return Ok(CommandInterpretation {
                intent: function_name.clone(),
                parameters,
                response_text: response.content,
                confidence,
                action,
                route,
                function_name: Some(function_name),
                usage: response.usage,
            });
        }

        if let Ok(json) = serde_json::from_str::<Value>(&response.content) {
            let intent = json
                .get("intent")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string();
            let parameters = json.get("parameters").cloned().unwrap_or_else(|| json!({}));
            let confidence = json
                .get("confidence")
                .and_then(Value::as_f64)
                .unwrap_or(0.5) as f32;
            let action = json
                .get("action")
                .and_then(Value::as_str)
                .map(|s| s.to_string());
            let route = json
                .get("route")
                .and_then(Value::as_str)
                .map(|s| s.to_string());
            let response_text = json
                .get("response")
                .and_then(Value::as_str)
                .unwrap_or(&response.content)
                .to_string();

            return Ok(CommandInterpretation {
                intent,
                parameters,
                response_text,
                confidence,
                action,
                route,
                function_name: None,
                usage: response.usage,
            });
        }
        Ok(CommandInterpretation {
            intent: "chat".to_string(),
            parameters: json!({}),
            response_text: response.content,
            confidence: 0.5,
            action: None,
            route: None,
            function_name: None,
            usage: response.usage,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandInterpretation {
    pub intent: String,
    pub parameters: Value,
    pub response_text: String,
    pub confidence: f32,
    pub action: Option<String>,
    pub route: Option<String>,
    pub function_name: Option<String>,
    pub usage: Usage,
}