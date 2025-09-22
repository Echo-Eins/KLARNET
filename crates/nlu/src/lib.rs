use klarnet_core::{CommandType, KlarnetResult, LocalCommand, NluResult, Transcript};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NluConfig {
    pub mode: NluMode,
    pub wake_words: Vec<String>,
}

impl Default for NluConfig {
    fn default() -> Self {
        Self {
            mode: NluMode::Local,
            wake_words: vec!["кларнет".to_string()],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NluMode {
    Local,
    Llm,
    Hybrid,
}

pub struct NluEngine {
    config: NluConfig,
}

impl NluEngine {
    pub async fn new(config: NluConfig) -> KlarnetResult<Self> {
        Ok(Self { config })
    }

    pub async fn process(&self, transcript: &Transcript) -> KlarnetResult<NluResult> {
        let text = transcript.full_text.to_lowercase();
        let wake_word_detected = self
            .config
            .wake_words
            .iter()
            .any(|wake| text.contains(wake));

        if !wake_word_detected && self.config.mode != NluMode::Llm {
            return Ok(NluResult {
                transcript: transcript.full_text.clone(),
                intent: None,
                wake_word_detected: false,
                command_type: CommandType::Unknown,
            });
        }

        let mut parameters = serde_json::Map::new();
        if let Some(app) = text.split_whitespace().nth(1) {
            parameters.insert(
                "app_name".to_string(),
                serde_json::Value::String(app.to_string()),
            );
        }

        let command = LocalCommand {
            action: "system.open_app".to_string(),
            parameters,
        };

        Ok(NluResult {
            transcript: transcript.full_text.clone(),
            intent: None,
            wake_word_detected,
            command_type: CommandType::Local(command),
        })
    }
}
