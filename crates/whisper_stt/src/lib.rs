use klarnet_core::{AudioChunk, KlarnetResult, Transcript, TranscriptSegment, WordInfo};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperConfig {
    pub language: String,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            language: "ru".to_string(),
        }
    }
}

pub struct WhisperEngine {
    config: WhisperConfig,
    metrics: WhisperMetrics,
}

#[derive(Debug, Default, Clone)]
pub struct WhisperMetrics {
    pub total_processed: u64,
}

impl WhisperEngine {
    pub async fn new(config: WhisperConfig) -> KlarnetResult<Self> {
        Ok(Self {
            config,
            metrics: WhisperMetrics::default(),
        })
    }

    pub async fn transcribe(&mut self, chunk: AudioChunk) -> KlarnetResult<Transcript> {
        self.metrics.total_processed += 1;
        let duration = chunk.total_samples as f64 / 16_000.0;
        let text = "симулированная команда".to_string();

        let segment = TranscriptSegment {
            start: 0.0,
            end: duration,
            text: text.clone(),
            confidence: 0.9,
            words: vec![WordInfo {
                word: text.clone(),
                start: 0.0,
                end: duration,
                confidence: 0.9,
            }],
        };

        Ok(Transcript {
            id: Uuid::new_v4(),
            language: self.config.language.clone(),
            segments: vec![segment],
            full_text: text,
            processing_time: Duration::from_millis(10),
        })
    }

    pub fn get_metrics(&self) -> WhisperMetrics {
        self.metrics.clone()
    }
}
