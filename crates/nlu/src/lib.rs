use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use klarnet_core::{
    CommandRoute, CommandRouting, CommandType, Entity, Intent, KlarnetError, KlarnetResult,
    LocalCommand, NluResult, Transcript,
};
use llm_connector::{
    CompletionRequest, CompletionResponse, Function as LlmFunction,
    LlmConfig as ConnectorLlmConfig, LlmConnector, LlmMetricsSnapshot, LlmProviderKind,
    Message as LlmMessage, Role as LlmRole, Usage,
};

use regex::{Regex, RegexBuilder};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Map as JsonMap, Value};
use tokio::sync::{Mutex as AsyncMutex, OwnedSemaphorePermit, Semaphore};
use tracing::{debug, warn};

const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.75;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NluConfig {
    pub mode: NluMode,
    #[serde(default = "default_wake_words")]
    pub wake_words: Vec<String>,
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,
    #[serde(default)]
    pub local: Option<LocalNluConfig>,
    #[serde(default)]
    pub llm: Option<LlmModeConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NluMode {
    Local,
    Llm,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalNluConfig {
    pub intents_path: PathBuf,
    pub entities_path: PathBuf,
    #[serde(default)]
    pub fallback: Option<FallbackConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FallbackConfig {
    #[serde(default)]
    pub action: Option<String>,
    #[serde(default)]
    pub route: Option<String>,
    #[serde(default)]
    pub parameters: JsonMap<String, Value>,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmModeConfig {
    #[serde(default = "default_llm_provider")]
    pub provider: String,
    pub api_key_env: String,
    pub model: String,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default = "default_llm_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_llm_temperature")]
    pub temperature: f32,
    #[serde(default = "default_llm_top_p")]
    pub top_p: f32,
    #[serde(default = "default_llm_timeout_s")]
    pub timeout_s: u64,
    #[serde(default = "default_llm_retry_attempts")]
    pub retry_attempts: u32,
    #[serde(default = "default_llm_cache_enabled")]
    pub cache_enabled: bool,
    #[serde(default = "default_llm_cache_ttl_s")]
    pub cache_ttl_s: u64,
    #[serde(default = "default_llm_max_concurrent_requests")]
    pub max_concurrent_requests: usize,
    #[serde(default = "default_llm_min_request_interval_ms")]
    pub min_request_interval_ms: u64,
}

impl Default for NluConfig {
    fn default() -> Self {
        Self {
            mode: NluMode::Local,
            wake_words: default_wake_words(),
            confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
            local: Some(LocalNluConfig::default()),
            llm: None,
        }
    }
}

impl Default for LocalNluConfig {
    fn default() -> Self {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            intents_path: base.join("../../config/patterns.yaml"),
            entities_path: base.join("../../config/entities.yaml"),
            fallback: None,
        }
    }
}

impl Default for LlmModeConfig {
    fn default() -> Self {
        Self {
            provider: default_llm_provider(),
            api_key_env: "OPENROUTER_API_KEY".to_string(),
            model: "x-ai/grok-4-fast:free".to_string(),
            base_url: None,
            max_tokens: default_llm_max_tokens(),
            temperature: default_llm_temperature(),
            top_p: default_llm_top_p(),
            timeout_s: default_llm_timeout_s(),
            retry_attempts: default_llm_retry_attempts(),
            cache_enabled: default_llm_cache_enabled(),
            cache_ttl_s: default_llm_cache_ttl_s(),
            max_concurrent_requests: default_llm_max_concurrent_requests(),
            min_request_interval_ms: default_llm_min_request_interval_ms(),
        }
    }
}

impl LlmModeConfig {
    fn to_connector_config(&self) -> KlarnetResult<ConnectorLlmConfig> {
        let provider = match self.provider.to_lowercase().as_str() {
            "openrouter" => LlmProviderKind::OpenRouter,
            "deepseek" => LlmProviderKind::DeepSeek,
            other => {
                return Err(KlarnetError::Nlu(format!(
                    "Unsupported LLM provider: {}",
                    other
                )))
            }
        };

        Ok(ConnectorLlmConfig {
            provider,
            model: self.model.clone(),
            api_key_env: self.api_key_env.clone(),
            base_url: self.base_url.clone(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            timeout_s: self.timeout_s,
            retry_attempts: self.retry_attempts,
            cache_enabled: self.cache_enabled,
            cache_ttl_s: self.cache_ttl_s,
        })
    }
}

fn default_wake_words() -> Vec<String> {
    vec!["кларнет".to_string()]
}

fn default_confidence_threshold() -> f32 {
    DEFAULT_CONFIDENCE_THRESHOLD
}

fn default_llm_provider() -> String {
    "openrouter".to_string()
}

fn default_llm_max_tokens() -> usize {
    500
}

fn default_llm_temperature() -> f32 {
    0.3
}

fn default_llm_top_p() -> f32 {
    0.95
}

fn default_llm_timeout_s() -> u64 {
    10
}

fn default_llm_retry_attempts() -> u32 {
    3
}

fn default_llm_cache_enabled() -> bool {
    true
}

fn default_llm_cache_ttl_s() -> u64 {
    3600
}

fn default_llm_max_concurrent_requests() -> usize {
    1
}

fn default_llm_min_request_interval_ms() -> u64 {
    0
}


pub struct NluEngine {
    config: NluConfig,
    wake_words_lower: Vec<String>,
    local_matcher: Option<LocalIntentMatcher>,
    llm_runtime: Option<LlmRuntime>,
    fallback: FallbackConfig,
}

impl NluEngine {
    pub async fn new(config: NluConfig) -> KlarnetResult<Self> {
        let wake_words_lower = config
            .wake_words
            .iter()
            .map(|w| w.to_lowercase())
            .collect::<Vec<_>>();

        let local_matcher = if let Some(local_config) = config.local.clone() {
            Some(LocalIntentMatcher::new(&local_config)?)
        } else {
            None
        };

        let fallback = local_matcher
            .as_ref()
            .map(|matcher| matcher.fallback.clone())
            .or_else(|| config.local.as_ref().and_then(|c| c.fallback.clone()))
            .unwrap_or_default();

        let llm_runtime = if matches!(config.mode, NluMode::Llm | NluMode::Hybrid) {
            if let Some(llm_config) = config.llm.clone() {
                let connector_config = llm_config.to_connector_config()?;
                let connector = Arc::new(LlmConnector::new(connector_config).await?);
                let concurrency = llm_config.max_concurrent_requests.max(1);
                let semaphore = Arc::new(Semaphore::new(concurrency));
                let min_interval = if llm_config.min_request_interval_ms > 0 {
                    Some(Duration::from_millis(llm_config.min_request_interval_ms))
                } else {
                    None
                };

                Some(LlmRuntime::new(
                    connector,
                    semaphore,
                    min_interval,
                    llm_config,
                ))
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            config,
            wake_words_lower,
            local_matcher,
            llm_runtime,
            fallback,
        })
    }
    pub async fn take_last_llm_usage(&self) -> Option<LlmUsageRecord> {
        match &self.llm_runtime {
            Some(runtime) => runtime.take_usage().await,
            None => None,
        }
    }

    pub fn llm_metrics_snapshot(&self) -> Option<LlmMetricsSnapshot> {
        self.llm_runtime
            .as_ref()
            .map(|runtime| runtime.metrics_snapshot())
    }

    pub fn llm_configuration(&self) -> Option<LlmConfigurationSummary> {
        self.llm_runtime.as_ref().map(|runtime| runtime.summary())
    }

    pub async fn process(&self, transcript: &Transcript) -> KlarnetResult<NluResult> {
        let (wake_word_detected, normalized_command, original_command, offset) =
            self.extract_command_text(&transcript.full_text);

        let normalized_command_trimmed = normalized_command.trim();
        let original_command_trimmed = original_command.trim();

        match self.config.mode {
            NluMode::Local => {
                self.process_local(
                    transcript,
                    wake_word_detected,
                    normalized_command_trimmed,
                    original_command_trimmed,
                    offset,
                )
                    .await
            }
            NluMode::Llm => {
                self.process_llm(
                    transcript,
                    wake_word_detected,
                    normalized_command_trimmed,
                    original_command_trimmed,
                )
                    .await
            }
            NluMode::Hybrid => {
                self.process_hybrid(
                    transcript,
                    wake_word_detected,
                    normalized_command_trimmed,
                    original_command_trimmed,
                    offset,
                )
                    .await
            }
        }
    }

    async fn process_local(
        &self,
        transcript: &Transcript,
        wake_word_detected: bool,
        normalized_command: &str,
        original_command: &str,
        offset: usize,
    ) -> KlarnetResult<NluResult> {
        if !wake_word_detected {
            return Ok(self.fallback_result(
                transcript,
                false,
                normalized_command,
                "wake_word_missing",
            ));
        }

        if normalized_command.is_empty() {
            return Ok(self.fallback_result(transcript, true, normalized_command, "empty_command"));
        }

        if let Some(matcher) = &self.local_matcher {
            if let Some(outcome) = matcher.match_text(normalized_command, offset)? {
                return Ok(self.build_local_result(transcript, wake_word_detected, outcome));
            }
        }

        Ok(self.fallback_result(
            transcript,
            wake_word_detected,
            original_command,
            "no_local_match",
        ))
    }

    async fn process_llm(
        &self,
        transcript: &Transcript,
        wake_word_detected: bool,
        normalized_command: &str,
        original_command: &str,
    ) -> KlarnetResult<NluResult> {
        if !wake_word_detected {
            return Ok(self.fallback_result(
                transcript,
                false,
                normalized_command,
                "wake_word_missing",
            ));
        }

        if original_command.is_empty() {
            return Ok(self.fallback_result(transcript, true, normalized_command, "empty_command"));
        }

        match self
            .invoke_llm(transcript, normalized_command, original_command)
            .await
        {
            Ok(Some(interp)) => {
                return Ok(self.build_llm_result(transcript, wake_word_detected, interp));
            }
            Ok(None) => {
                warn!("LLM returned no actionable interpretation");
                return Ok(self.fallback_result(
                    transcript,
                    wake_word_detected,
                    original_command,
                    "llm_invalid_response",
                ));
            }
            Err(KlarnetError::Network(err)) => {
                warn!("LLM network error: {}", err);
                return Ok(self.fallback_result(
                    transcript,
                    wake_word_detected,
                    original_command,
                    "llm_network_error",
                ));
            }
            Err(err) => {
                warn!("LLM invocation failed: {}", err);
                return Ok(self.fallback_result(
                    transcript,
                    wake_word_detected,
                    original_command,
                    "llm_error",
                ));
            }
        }
    }

    async fn process_hybrid(
        &self,
        transcript: &Transcript,
        wake_word_detected: bool,
        normalized_command: &str,
        original_command: &str,
        offset: usize,
    ) -> KlarnetResult<NluResult> {
        if !wake_word_detected {
            return Ok(self.fallback_result(
                transcript,
                false,
                normalized_command,
                "wake_word_missing",
            ));
        }

        let mut low_confidence_match = None;

        if let Some(matcher) = &self.local_matcher {
            if let Some(outcome) = matcher.match_text(normalized_command, offset)? {
                if outcome.confidence >= self.config.confidence_threshold {
                    debug!("Using local intent {}", outcome.intent_name);
                    return Ok(self.build_local_result(transcript, wake_word_detected, outcome));
                }
                low_confidence_match = Some(outcome);
            }
        }

        match self
            .invoke_llm(transcript, normalized_command, original_command)
            .await
        {
            Ok(Some(interp)) => {
                debug!("Using LLM interpretation for command");
                return Ok(self.build_llm_result(transcript, wake_word_detected, interp));
            }
            Ok(None) => {}
            Err(KlarnetError::Network(err)) => {
                warn!("LLM network error: {}", err);
                return Ok(self.fallback_result(
                    transcript,
                    wake_word_detected,
                    original_command,
                    "llm_network_error",
                ));
            }
            Err(err) => {
                warn!("LLM invocation failed: {}", err);
            }
        }

        if let Some(outcome) = low_confidence_match {
            debug!(
                "Falling back to low confidence local intent {}",
                outcome.intent_name
            );
            return Ok(self.build_local_result(transcript, wake_word_detected, outcome));
        }

        Ok(self.fallback_result(transcript, wake_word_detected, original_command, "no_match"))
    }

    fn build_local_result(
        &self,
        transcript: &Transcript,
        wake_word_detected: bool,
        outcome: MatchOutcome,
    ) -> NluResult {
        let action = outcome
            .action
            .clone()
            .unwrap_or_else(|| outcome.intent_name.clone());

        let routing_target = outcome.route.clone().or_else(|| Some(action.clone()));

        let mut parameters = outcome.parameters.clone();
        parameters.insert(
            "transcript".to_string(),
            Value::String(transcript.full_text.clone()),
        );

        NluResult {
            transcript: transcript.full_text.clone(),
            intent: Some(Intent {
                name: outcome.intent_name.clone(),
                confidence: outcome.confidence,
                entities: outcome.entities,
            }),
            wake_word_detected,
            command_type: CommandType::Local(LocalCommand {
                action: action.clone(),
                parameters: parameters.clone(),
            }),
            confidence: outcome.confidence,
            parameters,
            routing: CommandRouting {
                route: CommandRoute::Local,
                target: routing_target,
                reason: Some("pattern_match".to_string()),
            },
        }
    }

    fn build_llm_result(
        &self,
        transcript: &Transcript,
        wake_word_detected: bool,
        interpretation: LlmInterpretation,
    ) -> NluResult {
        let mut parameters = interpretation.parameters.clone();
        parameters.insert(
            "transcript".to_string(),
            Value::String(transcript.full_text.clone()),
        );

        if let Some(response) = interpretation.response_text.as_ref() {
            parameters
                .entry("llm_response".to_string())
                .or_insert_with(|| Value::String(response.clone()));
        }

        if let Some(function) = interpretation.function_name.as_ref() {
            parameters
                .entry("llm_function".to_string())
                .or_insert_with(|| Value::String(function.clone()));
        }

        let usage = &interpretation.usage;
        parameters
            .entry("llm_prompt_tokens".to_string())
            .or_insert_with(|| Value::Number(serde_json::Number::from(usage.prompt_tokens as u64)));
        parameters
            .entry("llm_completion_tokens".to_string())
            .or_insert_with(|| {
                Value::Number(serde_json::Number::from(usage.completion_tokens as u64))
            });
        parameters
            .entry("llm_total_tokens".to_string())
            .or_insert_with(|| Value::Number(serde_json::Number::from(usage.total_tokens as u64)));

        let command_type = if let Some(action) = &interpretation.action {
            CommandType::LlmRequired(action.clone())
        } else {
            CommandType::Unknown
        };

        NluResult {
            transcript: transcript.full_text.clone(),
            intent: interpretation.intent_name.map(|name| Intent {
                name,
                confidence: interpretation.confidence,
                entities: interpretation.entities,
            }),
            wake_word_detected,
            command_type,
            confidence: interpretation.confidence,
            parameters,
            routing: CommandRouting {
                route: CommandRoute::Llm,
                target: interpretation.route.or(interpretation.action),
                reason: Some("llm_match".to_string()),
            },
        }
    }

    fn fallback_result(
        &self,
        transcript: &Transcript,
        wake_word_detected: bool,
        command_text: &str,
        reason: &str,
    ) -> NluResult {
        let mut parameters = self.fallback.parameters.clone();
        if !command_text.is_empty() {
            parameters.insert("query".to_string(), Value::String(command_text.to_string()));
        }

        let reason_text = if let Some(base_reason) = &self.fallback.reason {
            format!("{}:{}", base_reason, reason)
        } else {
            reason.to_string()
        };

        let command_type = if let Some(action) = &self.fallback.action {
            CommandType::LlmRequired(action.clone())
        } else {
            CommandType::Unknown
        };

        NluResult {
            transcript: transcript.full_text.clone(),
            intent: None,
            wake_word_detected,
            command_type,
            confidence: 0.0,
            parameters,
            routing: CommandRouting {
                route: CommandRoute::Fallback,
                target: self.fallback.route.clone(),
                reason: Some(reason_text),
            },
        }
    }

    fn extract_command_text(&self, text: &str) -> (bool, String, String, usize) {
        if text.trim().is_empty() {
            return (false, String::new(), String::new(), text.len());
        }

        let trimmed_original = text.trim_start();
        let trimmed_lower = trimmed_original.to_lowercase();

        for wake in &self.wake_words_lower {
            if trimmed_lower.starts_with(wake) {
                let wake_chars = wake.chars().count();
                if !is_boundary(&trimmed_lower, wake_chars) {
                    continue;
                }

                let wake_bytes = char_offset(trimmed_original, wake_chars);
                let remainder = &trimmed_original[wake_bytes..];
                let remainder = remainder.trim_start_matches(|c: char| {
                    c.is_whitespace() || matches!(c, ',' | ':' | ';' | '-' | '–')
                });
                let offset = text.len().saturating_sub(remainder.len());
                let normalized = remainder.to_lowercase();
                return (true, normalized, remainder.to_string(), offset);
            }
        }

        let offset = text.len().saturating_sub(trimmed_original.len());
        (false, trimmed_lower, trimmed_original.to_string(), offset)
    }

    async fn invoke_llm(
        &self,
        transcript: &Transcript,
        normalized_command: &str,
        original_command: &str,
    ) -> KlarnetResult<Option<LlmInterpretation>> {
        let runtime = match &self.llm_runtime {
            Some(runtime) => runtime,
            None => return Ok(None),
        };

        let _permit = runtime.acquire().await;

        let request = self.build_llm_request(transcript, normalized_command, original_command);
        let start = Instant::now();
        let response = runtime.connector.complete(request).await?;
        let latency = start.elapsed();

        runtime.record_usage(response.usage.clone(), latency).await;

        match self.parse_llm_response(response)? {
            Some(mut interpretation) => {
                if self.validate_llm_interpretation(&mut interpretation) {
                    Ok(Some(interpretation))
                } else {
                    warn!("LLM interpretation rejected by validator");
                    Ok(None)
                }
            }
            None => Ok(None),
        }
    }
    fn build_llm_request(
        &self,
        transcript: &Transcript,
        normalized_command: &str,
        original_command: &str,
    ) -> CompletionRequest {
        let llm_config = self
            .config
            .llm
            .as_ref()
            .expect("LLM configuration missing for enabled mode");

        let system_prompt = format!(
            "You are KLARNET's NLU module. Extract intents, actions, routes and structured parameters. \
            Use function calls when appropriate. Known wake words: {}. Return Russian text when needed.",
            self.config.wake_words.join(", ")
        );

        let context = format!(
            "Original transcript: {original}\nNormalised: {normalized}\nLanguage: {language}",
            original = original_command.trim(),
            normalized = normalized_command.trim(),
            language = transcript.language,
        );

        CompletionRequest {
            messages: vec![
                LlmMessage {
                    role: LlmRole::System,
                    content: system_prompt,
                },
                LlmMessage {
                    role: LlmRole::User,
                    content: context,
                },
            ],
            max_tokens: Some(llm_config.max_tokens),
            temperature: Some(llm_config.temperature),
            top_p: Some(llm_config.top_p),
            stop: None,
            functions: Some(self.llm_functions()),
        }
    }

    fn llm_functions(&self) -> Vec<LlmFunction> {
        vec![
            LlmFunction {
                name: "execute_command".to_string(),
                description: "Execute a system, device or smart home action".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "intent": {"type": "string"},
                        "action": {"type": "string"},
                        "route": {"type": "string"},
                        "parameters": {"type": "object"},
                        "entities": {"type": "object"},
                        "response": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                    },
                    "required": ["action"]
                }),
            },
            LlmFunction {
                name: "answer_question".to_string(),
                description: "Provide an answer when no actionable command is required".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "intent": {"type": "string"},
                        "answer": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                    },
                    "required": ["answer"]
                }),
            },
        ]
    }

    fn parse_llm_response(
        &self,
        response: CompletionResponse,
    ) -> KlarnetResult<Option<LlmInterpretation>> {
        if let Some(function_call) = response.function_call.clone() {
            let args: Value = match serde_json::from_str(&function_call.arguments) {
                Ok(value) => value,
                Err(err) => {
                    warn!("Failed to parse LLM function arguments: {}", err);
                    return Ok(None);
                }
            };

            return Ok(self.parse_function_call(function_call.name, args, response));
        }

        if response.content.trim().is_empty() {
            return Ok(None);
        }

        match serde_json::from_str::<Value>(&response.content) {
            Ok(json) => Ok(self.parse_structured_content(json, response.usage)),
            Err(_) => Ok(Some(LlmInterpretation {
                intent_name: Some("chat".to_string()),
                confidence: 0.5,
                parameters: JsonMap::new(),
                entities: Vec::new(),
                action: None,
                route: None,
                response_text: Some(response.content),
                function_name: None,
                usage: response.usage,
            })),
        }
    }

    fn parse_function_call(
        &self,
        name: String,
        args: Value,
        response: CompletionResponse,
    ) -> Option<LlmInterpretation> {
        match name.as_str() {
            "execute_command" => {
                let action = args
                    .get("action")
                    .and_then(Value::as_str)?
                    .trim()
                    .to_string();
                if action.is_empty() {
                    return None;
                }

                let mut parameters = args
                    .get("parameters")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_else(JsonMap::new);
                let entities = self.extract_entities(&mut parameters, args.get("entities"));
                let intent_name = args
                    .get("intent")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string())
                    .or_else(|| Some(action.clone()));

                Some(LlmInterpretation {
                    intent_name,
                    confidence: args
                        .get("confidence")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.8) as f32,
                    parameters,
                    entities,
                    action: Some(action),
                    route: args
                        .get("route")
                        .and_then(Value::as_str)
                        .map(|s| s.to_string()),
                    response_text: args
                        .get("response")
                        .and_then(Value::as_str)
                        .map(|s| s.to_string())
                        .or_else(|| {
                            if response.content.trim().is_empty() {
                                None
                            } else {
                                Some(response.content.clone())
                            }
                        }),
                    function_name: Some(name),
                    usage: response.usage,
                })
            }
            "answer_question" => Some(LlmInterpretation {
                intent_name: args
                    .get("intent")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string())
                    .or_else(|| Some("chat".to_string())),
                confidence: args
                    .get("confidence")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.6) as f32,
                parameters: JsonMap::new(),
                entities: Vec::new(),
                action: None,
                route: None,
                response_text: args
                    .get("answer")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string())
                    .or_else(|| Some(response.content)),
                function_name: Some(name),
                usage: response.usage,
            }),
            _ => None,
        }
    }

    fn parse_structured_content(&self, json: Value, usage: Usage) -> Option<LlmInterpretation> {
        let mut parameters = json
            .get("parameters")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_else(JsonMap::new);

        let entities = self.extract_entities(&mut parameters, json.get("entities"));

        Some(LlmInterpretation {
            intent_name: json
                .get("intent")
                .and_then(Value::as_str)
                .map(|s| s.to_string()),
            confidence: json
                .get("confidence")
                .and_then(Value::as_f64)
                .unwrap_or(0.5) as f32,
            parameters,
            entities,
            action: json
                .get("action")
                .and_then(Value::as_str)
                .map(|s| s.to_string()),
            route: json
                .get("route")
                .and_then(Value::as_str)
                .map(|s| s.to_string())
                .or_else(|| {
                    json.get("routing")
                        .and_then(|r| r.get("target"))
                        .and_then(Value::as_str)
                        .map(|s| s.to_string())
                }),
            response_text: json
                .get("response")
                .and_then(Value::as_str)
                .map(|s| s.to_string()),
            function_name: None,
            usage,
        })
    }

    fn extract_entities(
        &self,
        parameters: &mut JsonMap<String, Value>,
        explicit: Option<&Value>,
    ) -> Vec<Entity> {
        let mut entities = Vec::new();

        if let Some(value) = parameters.remove("entities") {
            if let Value::Object(map) = value {
                for (name, value) in map {
                    entities.push(Entity {
                        name,
                        value,
                        start: 0,
                        end: 0,
                    });
                }
            }
        }

        if let Some(Value::Object(map)) = explicit {
            for (name, value) in map.clone() {
                entities.push(Entity {
                    name,
                    value,
                    start: 0,
                    end: 0,
                });
            }
        }

        entities
    }

    fn validate_llm_interpretation(&self, interpretation: &mut LlmInterpretation) -> bool {
        if interpretation.confidence.is_nan() {
            interpretation.confidence = 0.0;
        }
        interpretation.confidence = interpretation.confidence.clamp(0.0, 1.0);

        if let Some(action) = &interpretation.action {
            if action.trim().is_empty() {
                interpretation.action = None;
            }
        }

        if interpretation.action.is_some() && interpretation.intent_name.is_none() {
            interpretation.intent_name = interpretation.action.clone();
        }

        true
    }
}

fn is_boundary(text: &str, chars: usize) -> bool {
    match text.chars().nth(chars) {
        Some(c) => !c.is_alphanumeric(),
        None => true,
    }
}

fn char_offset(text: &str, chars: usize) -> usize {
    if chars == 0 {
        return 0;
    }
    match text.char_indices().nth(chars) {
        Some((idx, _)) => idx,
        None => text.len(),
    }
}

struct LlmRuntime {
    connector: Arc<LlmConnector>,
    semaphore: Arc<Semaphore>,
    min_interval: Option<Duration>,
    last_call: AsyncMutex<Option<Instant>>,
    last_usage: AsyncMutex<Option<LlmUsageRecord>>,
    config: LlmModeConfig,
}

impl LlmRuntime {
    fn new(
        connector: Arc<LlmConnector>,
        semaphore: Arc<Semaphore>,
        min_interval: Option<Duration>,
        config: LlmModeConfig,
    ) -> Self {
        Self {
            connector,
            semaphore,
            min_interval,
            last_call: AsyncMutex::new(None),
            last_usage: AsyncMutex::new(None),
            config,
        }
    }

    async fn acquire(&self) -> OwnedSemaphorePermit {
        let permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("LLM semaphore closed");

        if let Some(interval) = self.min_interval {
            let mut last_call = self.last_call.lock().await;
            if let Some(last) = *last_call {
                let elapsed = last.elapsed();
                if elapsed < interval {
                    tokio::time::sleep(interval - elapsed).await;
                }
            }
            *last_call = Some(Instant::now());
        }

        permit
    }

    async fn record_usage(&self, usage: Usage, latency: Duration) {
        let config = self.connector.config();
        let provider = match &config.provider {
            LlmProviderKind::OpenRouter => "openrouter".to_string(),
            LlmProviderKind::DeepSeek => "deepseek".to_string(),
            LlmProviderKind::OpenAI => "openai".to_string(),
            LlmProviderKind::Custom(name) => name.clone(),
        };

        let mut guard = self.last_usage.lock().await;
        *guard = Some(LlmUsageRecord {
            usage,
            latency,
            provider,
            model: config.model.clone(),
        });
    }

    async fn take_usage(&self) -> Option<LlmUsageRecord> {
        let mut guard = self.last_usage.lock().await;
        guard.take()
    }

    fn metrics_snapshot(&self) -> LlmMetricsSnapshot {
        self.connector.metrics_snapshot()
    }

    fn summary(&self) -> LlmConfigurationSummary {
        LlmConfigurationSummary {
            provider: self.config.provider.clone(),
            model: self.config.model.clone(),
            cache_enabled: self.config.cache_enabled,
            max_concurrent_requests: self.config.max_concurrent_requests,
            min_request_interval_ms: self.config.min_request_interval_ms,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LlmUsageRecord {
    pub usage: Usage,
    pub latency: Duration,
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Clone)]
pub struct LlmConfigurationSummary {
    pub provider: String,
    pub model: String,
    pub cache_enabled: bool,
    pub max_concurrent_requests: usize,
    pub min_request_interval_ms: u64,
}

#[derive(Debug)]
struct MatchOutcome {
    intent_name: String,
    confidence: f32,
    action: Option<String>,
    parameters: JsonMap<String, Value>,
    entities: Vec<Entity>,
    route: Option<String>,
}

#[derive(Debug)]
struct LocalIntentMatcher {
    intents: Vec<CompiledIntent>,
    fallback: FallbackConfig,
}

impl LocalIntentMatcher {
    fn new(config: &LocalNluConfig) -> KlarnetResult<Self> {
        let intents: IntentConfig = load_config(&config.intents_path)?;
        let entities: EntitiesConfig = load_config(&config.entities_path)?;

        let entity_matchers = entities.build_matchers()?;
        let file_fallback = intents.fallback.clone();
        let compiled = intents
            .intents
            .into_iter()
            .map(|intent| intent.compile(&entity_matchers))
            .collect::<KlarnetResult<Vec<_>>>()?;

        let fallback = config
            .fallback
            .clone()
            .or(file_fallback)
            .unwrap_or_default();

        Ok(Self {
            intents: compiled,
            fallback,
        })
    }

    fn match_text(&self, text: &str, offset: usize) -> KlarnetResult<Option<MatchOutcome>> {
        for intent in &self.intents {
            if let Some(outcome) = intent.match_text(text, offset)? {
                return Ok(Some(outcome));
            }
        }
        Ok(None)
    }
}

#[derive(Debug)]
struct CompiledIntent {
    name: String,
    confidence: f32,
    action: Option<String>,
    route: Option<String>,
    parameters: JsonMap<String, Value>,
    patterns: Vec<CompiledPattern>,
}

impl CompiledIntent {
    fn match_text(&self, text: &str, offset: usize) -> KlarnetResult<Option<MatchOutcome>> {
        for pattern in &self.patterns {
            if let Some(caps) = pattern.regex.captures(text) {
                let mut parameters = self.parameters.clone();
                let mut slot_values: HashMap<String, Value> = HashMap::new();
                let mut entities = Vec::new();

                for slot in &pattern.slots {
                    if let Some(m) = caps.name(&slot.name) {
                        let value = slot.matcher.resolve_value(m.as_str())?;
                        slot_values.insert(slot.name.clone(), value.clone());
                        parameters.insert(slot.name.clone(), value.clone());
                        entities.push(Entity {
                            name: slot.name.clone(),
                            value,
                            start: offset + m.start(),
                            end: offset + m.end(),
                        });
                    }
                }

                for (key, value) in self.parameters.iter() {
                    if let Some(slot_name) = value
                        .as_str()
                        .and_then(|v| v.strip_prefix('{'))
                        .and_then(|v| v.strip_suffix('}'))
                    {
                        if let Some(slot_value) = slot_values.get(slot_name) {
                            parameters.insert(key.clone(), slot_value.clone());
                        }
                    }
                }

                let outcome = MatchOutcome {
                    intent_name: self.name.clone(),
                    confidence: self.confidence,
                    action: self.action.clone(),
                    parameters,
                    entities,
                    route: self.route.clone(),
                };

                return Ok(Some(outcome));
            }
        }

        Ok(None)
    }
}

#[derive(Debug)]
struct CompiledPattern {
    regex: Regex,
    slots: Vec<SlotBinding>,
}

#[derive(Debug, Clone)]
struct SlotBinding {
    name: String,
    matcher: EntityMatcher,
}

#[derive(Debug, Clone)]
enum EntityMatcher {
    List {
        synonyms: HashMap<String, String>,
    },
    Regex {
        pattern: String,
        transform: Option<EntityTransform>,
    },
}

#[derive(Debug, Clone)]
enum EntityTransform {
    Int,
    Float,
    Lowercase,
}

impl EntityMatcher {
    fn regex_fragment(&self) -> KlarnetResult<String> {
        Ok(match self {
            EntityMatcher::List { synonyms } => {
                if synonyms.is_empty() {
                    return Err(KlarnetError::Nlu(
                        "Entity list must contain at least one value".to_string(),
                    ));
                }
                let mut variants = synonyms.keys().cloned().collect::<Vec<_>>();
                variants.sort_by(|a, b| b.len().cmp(&a.len()));
                let escaped = variants
                    .into_iter()
                    .map(|v| regex::escape(&v))
                    .collect::<Vec<_>>()
                    .join("|");
                format!("(?:{})", escaped)
            }
            EntityMatcher::Regex { pattern, .. } => format!("(?:{})", pattern),
        })
    }

    fn resolve_value(&self, raw: &str) -> KlarnetResult<Value> {
        match self {
            EntityMatcher::List { synonyms } => {
                let key = raw.trim().to_lowercase();
                if let Some(value) = synonyms.get(&key) {
                    Ok(Value::String(value.clone()))
                } else {
                    Err(KlarnetError::Nlu(format!("Unknown entity value '{}'", raw)))
                }
            }
            EntityMatcher::Regex { transform, .. } => {
                let trimmed = raw.trim();
                match transform {
                    Some(EntityTransform::Int) => trimmed
                        .parse::<i64>()
                        .map(|v| Value::Number(v.into()))
                        .map_err(|e| {
                            KlarnetError::Nlu(format!(
                                "Failed to parse integer value '{}': {}",
                                trimmed, e
                            ))
                        }),
                    Some(EntityTransform::Float) => trimmed
                        .parse::<f64>()
                        .map(|v| Value::Number(serde_json::Number::from_f64(v).unwrap()))
                        .map_err(|e| {
                            KlarnetError::Nlu(format!(
                                "Failed to parse float value '{}': {}",
                                trimmed, e
                            ))
                        }),
                    Some(EntityTransform::Lowercase) => Ok(Value::String(trimmed.to_lowercase())),
                    None => Ok(Value::String(trimmed.to_string())),
                }
            }
        }
    }
}

impl EntityTransform {
    fn from_str(value: &str) -> Option<Self> {
        match value {
            "int" => Some(EntityTransform::Int),
            "float" => Some(EntityTransform::Float),
            "lowercase" => Some(EntityTransform::Lowercase),
            _ => None,
        }
    }
}

#[derive(Debug, Deserialize)]
struct IntentConfig {
    intents: Vec<IntentDefinition>,
    #[serde(default)]
    fallback: Option<FallbackConfig>,
}

#[derive(Debug, Deserialize)]
struct IntentDefinition {
    name: String,
    #[serde(default)]
    patterns: Vec<String>,
    #[serde(default = "default_intent_confidence")]
    confidence: f32,
    #[serde(default)]
    action: Option<String>,
    #[serde(default)]
    parameters: JsonMap<String, Value>,
    #[serde(default)]
    route: Option<String>,
}

fn default_intent_confidence() -> f32 {
    0.8
}

impl IntentDefinition {
    fn compile(self, entities: &HashMap<String, EntityMatcher>) -> KlarnetResult<CompiledIntent> {
        let placeholder_regex = Regex::new(r"\{([a-zA-Z0-9_]+)\}").unwrap();
        let mut patterns = Vec::new();

        for pattern in &self.patterns {
            let mut result = String::new();
            let mut slots = Vec::new();
            let mut last = 0;

            for caps in placeholder_regex.captures_iter(pattern) {
                let m = caps.get(0).unwrap();
                result.push_str(&pattern[last..m.start()]);
                let slot_name = caps.get(1).unwrap().as_str().to_string();
                let matcher = entities.get(&slot_name).ok_or_else(|| {
                    KlarnetError::Nlu(format!(
                        "Unknown entity '{}' referenced in intent '{}'",
                        slot_name, self.name
                    ))
                })?;
                result.push_str(&format!("(?P<{}>{})", slot_name, matcher.regex_fragment()?));
                slots.push(SlotBinding {
                    name: slot_name,
                    matcher: matcher.clone(),
                });
                last = m.end();
            }

            result.push_str(&pattern[last..]);

            let regex = RegexBuilder::new(&format!(r"^\s*{}\s*$", result))
                .case_insensitive(true)
                .unicode(true)
                .build()
                .map_err(|e| {
                    KlarnetError::Nlu(format!(
                        "Invalid pattern '{}' for intent '{}': {}",
                        pattern, self.name, e
                    ))
                })?;

            patterns.push(CompiledPattern { regex, slots });
        }

        Ok(CompiledIntent {
            name: self.name,
            confidence: self.confidence,
            action: self.action,
            route: self.route,
            parameters: self.parameters,
            patterns,
        })
    }
}


#[derive(Debug, Deserialize)]
struct EntitiesConfig {
    entities: HashMap<String, RawEntityDefinition>,
}

impl EntitiesConfig {
    fn build_matchers(&self) -> KlarnetResult<HashMap<String, EntityMatcher>> {
        let mut result = HashMap::new();

        for (name, definition) in &self.entities {
            result.insert(name.clone(), definition.to_matcher()?);
        }

        Ok(result)
    }
}

#[derive(Debug, Deserialize)]
struct RawEntityDefinition {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    values: Option<RawEntityValues>,
    #[serde(default)]
    pattern: Option<String>,
    #[serde(default)]
    transform: Option<String>,
}

impl RawEntityDefinition {
    fn to_matcher(&self) -> KlarnetResult<EntityMatcher> {
        match self.kind.as_str() {
            "list" => {
                let values = self.values.clone().ok_or_else(|| {
                    KlarnetError::Nlu("List entity must provide values".to_string())
                })?;
                let mut synonyms = HashMap::new();
                for (synonym, value) in values.into_pairs() {
                    synonyms.insert(synonym.to_lowercase(), value.clone());
                    synonyms.entry(value.to_lowercase()).or_insert(value);
                }
                Ok(EntityMatcher::List { synonyms })
            }
            "regex" => {
                let pattern = self.pattern.clone().ok_or_else(|| {
                    KlarnetError::Nlu("Regex entity must provide pattern".to_string())
                })?;
                let transform = self
                    .transform
                    .as_deref()
                    .and_then(EntityTransform::from_str);
                Ok(EntityMatcher::Regex { pattern, transform })
            }
            other => Err(KlarnetError::Nlu(format!(
                "Unsupported entity type '{}'",
                other
            ))),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum RawEntityValues {
    List(Vec<HashMap<String, String>>),
    Map(HashMap<String, String>),
}

impl RawEntityValues {
    fn into_pairs(self) -> Vec<(String, String)> {
        match self {
            RawEntityValues::List(list) => list
                .into_iter()
                .flat_map(|entry| entry.into_iter())
                .collect(),
            RawEntityValues::Map(map) => map.into_iter().collect(),
        }
    }
}

fn load_config<T: DeserializeOwned>(path: &Path) -> KlarnetResult<T> {
    let contents = fs::read_to_string(path)
        .map_err(|e| KlarnetError::Config(format!("Failed to read {}: {}", path.display(), e)))?;

    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");

    if ext.eq_ignore_ascii_case("json") {
        serde_json::from_str(&contents)
            .map_err(|e| KlarnetError::Config(format!("Failed to parse {}: {}", path.display(), e)))
    } else {
        serde_yaml::from_str(&contents)
            .map_err(|e| KlarnetError::Config(format!("Failed to parse {}: {}", path.display(), e)))
    }
}

struct LlmInterpretation {
    intent_name: Option<String>,
    confidence: f32,
    parameters: JsonMap<String, Value>,
    entities: Vec<Entity>,
    action: Option<String>,
    route: Option<String>,
    response_text: Option<String>,
    function_name: Option<String>,
    usage: Usage,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, time::Duration};
    use tempfile::tempdir;
    use uuid::Uuid;

    fn transcript(text: &str) -> Transcript {
        Transcript {
            id: Uuid::new_v4(),
            language: "ru".to_string(),
            segments: Vec::new(),
            full_text: text.to_string(),
            processing_time: Duration::from_millis(0),
        }
    }

    #[tokio::test]
    async fn detects_wake_word_and_matches_light_intent() {
        let engine = NluEngine::new(NluConfig::default()).await.unwrap();
        let transcript = transcript("Кларнет включи свет");
        let result = engine.process(&transcript).await.unwrap();

        assert!(result.wake_word_detected);
        assert_eq!(result.intent.as_ref().unwrap().name, "lights_on");
        let route = result.routing.route.clone();
        assert!(matches!(route, CommandRoute::Local));
        assert_eq!(
            result
                .parameters
                .get("state")
                .and_then(Value::as_str)
                .unwrap(),
            "on"
        );
    }

    #[tokio::test]
    async fn extracts_entities_from_slots() {
        let engine = NluEngine::new(NluConfig::default()).await.unwrap();
        let transcript = transcript("Кларнет поставь таймер на 10 минут");
        let result = engine.process(&transcript).await.unwrap();

        let params = &result.parameters;
        assert_eq!(result.intent.as_ref().unwrap().name, "set_timer");
        assert_eq!(params.get("number").unwrap().as_i64().unwrap(), 10);
        assert_eq!(
            params.get("time_unit").and_then(Value::as_str).unwrap(),
            "minutes"
        );
    }

    #[tokio::test]
    async fn handles_phrases_without_wake_word() {
        let engine = NluEngine::new(NluConfig::default()).await.unwrap();
        let transcript = transcript("включи свет");
        let result = engine.process(&transcript).await.unwrap();

        assert!(!result.wake_word_detected);
        assert!(result.intent.is_none());
        let route = result.routing.route.clone();
        assert!(matches!(route, CommandRoute::Fallback));
        assert!(matches!(result.command_type, CommandType::Unknown));
    }

    #[tokio::test]
    async fn falls_back_for_unknown_intent() {
        let engine = NluEngine::new(NluConfig::default()).await.unwrap();
        let transcript = transcript("Кларнет расскажи анекдот");
        let result = engine.process(&transcript).await.unwrap();

        assert!(result.wake_word_detected);
        assert!(result.intent.is_none());
        let route = result.routing.route.clone();
        assert!(matches!(route, CommandRoute::Fallback));
    }

    #[tokio::test]
    async fn loads_configuration_from_json() {
        let dir = tempdir().unwrap();
        let intents_path = dir.path().join("intents.json");
        let entities_path = dir.path().join("entities.json");

        fs::write(
            &intents_path,
            serde_json::json!({
                "intents": [
                    {
                        "name": "greet",
                        "patterns": ["привет"],
                        "confidence": 0.6,
                        "action": "assistant.greet",
                        "parameters": {"response": "hello"}
                    }
                ]
            })
                .to_string(),
        )
            .unwrap();

        fs::write(
            &entities_path,
            serde_json::json!({"entities": {}}).to_string(),
        )
            .unwrap();

        let mut config = NluConfig::default();
        config.mode = NluMode::Local;
        config.wake_words = vec!["ассистент".to_string()];
        config.local = Some(LocalNluConfig {
            intents_path: intents_path.clone(),
            entities_path: entities_path.clone(),
            fallback: None,
        });

        let engine = NluEngine::new(config).await.unwrap();
        let transcript = transcript("Ассистент привет");
        let result = engine.process(&transcript).await.unwrap();

        assert!(result.wake_word_detected);
        assert_eq!(result.intent.as_ref().unwrap().name, "greet");
        assert_eq!(
            result
                .parameters
                .get("response")
                .and_then(Value::as_str)
                .unwrap(),
            "hello"
        );
    }
}