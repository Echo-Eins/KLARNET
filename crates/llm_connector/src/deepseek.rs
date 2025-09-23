use std::time::Duration;

use async_trait::async_trait;
use klarnet_core::{KlarnetError, KlarnetResult};
use reqwest::Client;
use serde_json::{json, Value};

use crate::{CompletionRequest, CompletionResponse, FunctionCall, LlmConfig, LlmProvider, Usage};

pub struct DeepSeekProvider {
    config: LlmConfig,
    client: Client,
    api_key: String,
}

impl DeepSeekProvider {
    pub async fn new(config: LlmConfig) -> KlarnetResult<Self> {
        let api_key = std::env::var(&config.api_key_env)
            .map_err(|_| KlarnetError::Nlu(format!("API key not found: {}", config.api_key_env)))?;

        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_s))
            .build()
            .map_err(|err| KlarnetError::Network(format!("failed to build client: {err}")))?;

        Ok(Self {
            config,
            client,
            api_key,
        })
    }

    fn endpoint(&self) -> String {
        self.config
            .base_url
            .as_ref()
            .cloned()
            .unwrap_or_else(|| "https://api.deepseek.com/chat/completions".to_string())
    }
}

#[async_trait]
impl LlmProvider for DeepSeekProvider {
    async fn complete(&self, request: CompletionRequest) -> KlarnetResult<CompletionResponse> {
        let mut payload = json!({
            "model": self.config.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens.unwrap_or(self.config.max_tokens),
            "temperature": request.temperature.unwrap_or(self.config.temperature),
            "top_p": request.top_p.unwrap_or(self.config.top_p),
        });

        if let Some(stop) = request.stop {
            payload["stop"] = json!(stop);
        }
        if let Some(functions) = request.functions {
            payload["functions"] = json!(functions);
        }

        let response = self
            .client
            .post(self.endpoint())
            .bearer_auth(&self.api_key)
            .json(&payload)
            .send()
            .await
            .map_err(|err| KlarnetError::Network(format!("LLM request failed: {err}")))?;

        let status = response.status();
        let json: Value = response
            .json()
            .await
            .map_err(|err| KlarnetError::Nlu(format!("Failed to parse LLM response: {err}")))?;

        if !status.is_success() {
            let message = json["error"]
                .get("message")
                .and_then(Value::as_str)
                .map(|s| s.to_string());
            return Err(KlarnetError::Nlu(format!(
                "DeepSeek returned {status}: {}",
                message.unwrap_or_else(|| json.to_string())
            )));
        }

        let message = json
            .pointer("/choices/0/message")
            .cloned()
            .ok_or_else(|| KlarnetError::Nlu("Missing message in DeepSeek response".to_string()))?;

        let content = message
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();

        let function_call = message.get("function_call").map(|call| FunctionCall {
            name: call
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            arguments: call
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        });

        let usage = Usage {
            prompt_tokens: json
                .get("usage")
                .and_then(|usage| usage.get("prompt_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize,
            completion_tokens: json
                .get("usage")
                .and_then(|usage| usage.get("completion_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize,
            total_tokens: json
                .get("usage")
                .and_then(|usage| usage.get("total_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize,
        };

        Ok(CompletionResponse {
            content,
            function_call,
            usage,
        })
    }

    fn name(&self) -> &str {
        "deepseek"
    }
}