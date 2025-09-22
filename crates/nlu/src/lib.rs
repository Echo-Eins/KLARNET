// crates/nlu/src/lib.rs

use async_trait::async_trait;
use klarnet_core::{
    CommandType, Entity, Intent, KlarnetError, KlarnetResult, LocalCommand, NluResult, Transcript,
};
use parking_lot::RwLock;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

pub mod patterns;
pub mod llm;
pub mod hybrid;

pub use patterns::PatternMatcher;
pub use llm::LlmProcessor;
pub use hybrid::HybridNlu;

/// NLU configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NluConfig {
    pub mode: NluMode,
    pub wake_words: Vec<String>,
    pub confidence_threshold: f32,
    pub patterns_file: Option<String>,
    pub llm_config: Option<LlmConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NluMode {
    Local,
    Llm,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: String,
    pub model: String,
    pub api_key_env: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub timeout_s: u64,
}

impl Default for NluConfig {
    fn default() -> Self {
        Self {
            mode: NluMode::Hybrid,
            wake_words: vec!["джарвис".to_string(), "ассистент".to_string()],
            confidence_threshold: 0.7,
            patterns_file: Some("config/patterns.yaml".to_string()),
            llm_config: Some(LlmConfig {
                provider: "openrouter".to_string(),
                model: "deepseek/deepseek-chat".to_string(),
                api_key_env: "OPENROUTER_API_KEY".to_string(),
                max_tokens: 500,
                temperature: 0.3,
                timeout_s: 5,
            }),
        }
    }
}

/// NLU Engine
pub struct NluEngine {
    config: NluConfig,
    processor: Box<dyn NluProcessor>,
    wake_word_detector: WakeWordDetector,
    metrics: Arc<RwLock<NluMetrics>>,
}

#[derive(Debug, Default)]
struct NluMetrics {
    total_processed: u64,
    wake_word_detections: u64,
    local_matches: u64,
    llm_requests: u64,
    avg_processing_time_ms: f64,
}

#[async_trait]
pub trait NluProcessor: Send + Sync {
    async fn process(&self, text: &str) -> KlarnetResult<NluResult>;
    fn name(&self) -> &str;
}

impl NluEngine {
    pub async fn new(config: NluConfig) -> KlarnetResult<Self> {
        let processor: Box<dyn NluProcessor> = match config.mode {
            NluMode::Local => {
                Box::new(PatternMatcher::new(config.patterns_file.clone()).await?)
            }
            NluMode::Llm => {
                Box::new(LlmProcessor::new(config.llm_config.clone()).await?)
            }
            NluMode::Hybrid => {
                Box::new(HybridNlu::new(config.clone()).await?)
            }
        };

        let wake_word_detector = WakeWordDetector::new(config.wake_words.clone());
        let metrics = Arc::new(RwLock::new(NluMetrics::default()));

        Ok(Self {
            config,
            processor,
            wake_word_detector,
            metrics,
        })
    }

    pub async fn process(&self, transcript: &Transcript) -> KlarnetResult<NluResult> {
        let text = transcript.full_text.to_lowercase();

        // Check for wake word
        let (wake_word_detected, command_text) = self.wake_word_detector.detect(&text);

        if !wake_word_detected && self.config.mode != NluMode::Llm {
            // No wake word and not in always-on LLM mode
            return Ok(NluResult {
                transcript: transcript.full_text.clone(),
                intent: None,
                wake_word_detected: false,
                command_type: CommandType::Unknown,
            });
        }

        // Process the command
        let mut result = self.processor.process(&command_text).await?;
        result.wake_word_detected = wake_word_detected;

        // Update metrics
        let mut metrics = self.metrics.write();
        metrics.total_processed += 1;
        if wake_word_detected {
            metrics.wake_word_detections += 1;
        }

        Ok(result)
    }
}

/// Wake word detector
struct WakeWordDetector {
    wake_words: Vec<String>,
    patterns: Vec<Regex>,
}

impl WakeWordDetector {
    fn new(wake_words: Vec<String>) -> Self {
        let patterns = wake_words
            .iter()
            .map(|w| Regex::new(&format!(r"\b{}\b", regex::escape(w))).unwrap())
            .collect();

        Self {
            wake_words,
            patterns,
        }
    }

    fn detect(&self, text: &str) -> (bool, String) {
        let lower_text = text.to_lowercase();

        for (i, pattern) in self.patterns.iter().enumerate() {
            if let Some(mat) = pattern.find(&lower_text) {
                let command_text = lower_text[mat.end()..].trim().to_string();
                debug!("Wake word '{}' detected", self.wake_words[i]);
                return (true, command_text);
            }
        }

        (false, text.to_string())
    }
}