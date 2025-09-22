// crates/nlu/src/llm.rs

use reqwest::Client;
use serde_json::json;
use std::time::Duration;

pub struct LlmProcessor {
    config: Option<LlmConfig>,
    client: Client,
    system_prompt: String,
}

impl LlmProcessor {
    pub async fn new(config: Option<LlmConfig>) -> KlarnetResult<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.as_ref().map(|c| c.timeout_s).unwrap_or(5)))
            .build()
            .map_err(|e| KlarnetError::Nlu(e.to_string()))?;

        let system_prompt = r#"
You are a voice assistant NLU processor. Extract intent and entities from user commands.
Respond in JSON format:
{
    "intent": "intent_name",
    "confidence": 0.0-1.0,
    "entities": {
        "entity_name": "value"
    },
    "action": "action_to_execute",
    "response": "optional response text"
}
Common intents: lights_control, open_app, set_timer, weather, music_play, smart_home, system_control
"#.to_string();

        Ok(Self {
            config,
            client,
            system_prompt,
        })
    }

    async fn call_llm(&self, text: &str) -> KlarnetResult<serde_json::Value> {
        let config = self.config.as_ref()
            .ok_or_else(|| KlarnetError::Nlu("LLM config not provided".to_string()))?;

        let api_key = std::env::var(&config.api_key_env)
            .map_err(|_| KlarnetError::Nlu(format!("API key not found: {}", config.api_key_env)))?;

        let response = self.client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&json!({
                "model": config.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ],
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
            }))
            .send()
            .await
            .map_err(|e| KlarnetError::Nlu(format!("LLM request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(KlarnetError::Nlu(format!("LLM error: {}", response.status())));
        }

        let json: serde_json::Value = response.json().await
            .map_err(|e| KlarnetError::Nlu(format!("Failed to parse LLM response: {}", e)))?;

        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| KlarnetError::Nlu("Invalid LLM response format".to_string()))?;

        serde_json::from_str(content)
            .map_err(|e| KlarnetError::Nlu(format!("Failed to parse LLM JSON: {}", e)))
    }
}

#[async_trait]
impl NluProcessor for LlmProcessor {
    async fn process(&self, text: &str) -> KlarnetResult<NluResult> {
        match self.call_llm(text).await {
            Ok(json) => {
                let intent = Intent {
                    name: json["intent"].as_str().unwrap_or("unknown").to_string(),
                    confidence: json["confidence"].as_f64().unwrap_or(0.5) as f32,
                    entities: Vec::new(), // Parse from json["entities"] if needed
                };

                let command_type = if let Some(action) = json["action"].as_str() {
                    CommandType::LlmRequired(action.to_string())
                } else {
                    CommandType::Unknown
                };

                Ok(NluResult {
                    transcript: text.to_string(),
                    intent: Some(intent),
                    wake_word_detected: false,
                    command_type,
                })
            }
            Err(e) => {
                warn!("LLM processing failed: {}", e);
                Ok(NluResult {
                    transcript: text.to_string(),
                    intent: None,
                    wake_word_detected: false,
                    command_type: CommandType::Unknown,
                })
            }
        }
    }

    fn name(&self) -> &str {
        "LlmProcessor"
    }
}