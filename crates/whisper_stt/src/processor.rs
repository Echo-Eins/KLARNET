// crates/whisper_stt/src/processor.rs
use std::time::Duration;

use klarnet_core::{KlarnetError, KlarnetResult, Transcript, TranscriptSegment, WordInfo};
use std::process::Stdio;
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tracing::info;
use uuid::Uuid;

use crate::{WhisperBackendConfig, WhisperConfig};

pub struct WhisperProcessor {
    config: WhisperConfig,
    process: Option<Child>,
    tx: Option<mpsc::UnboundedSender<Vec<f32>>>,
    rx: Option<mpsc::UnboundedReceiver<Transcript>>,
}

impl WhisperProcessor {
    pub async fn new(config: WhisperConfig) -> KlarnetResult<Self> {
        // In production, this would initialize the actual faster-whisper model
        // through Python bindings or C++ library
        Ok(Self {
            config,
            process: None,
            tx: None,
            rx: None,
        })
    }

    pub async fn start_process(&mut self) -> KlarnetResult<()> {
        let python_config = match &self.config.backend {
            WhisperBackendConfig::Python(cfg) => cfg,
            WhisperBackendConfig::Native => {
                return Err(KlarnetError::Stt(
                    "Native backend is not supported by WhisperProcessor".to_string(),
                ))
            }
        };
        // Start Python process for faster-whisper
        let mut cmd = Command::new(&python_config.executable);
        cmd.arg("-u")
            .arg(&python_config.script)
            .arg("--model-path")
            .arg(self.config.model.model_path.to_string_lossy().to_string())
            .arg("--language")
            .arg(&self.config.language)
            .arg("--compute-type")
            .arg(&self.config.model.compute_type)
            .arg("--device")
            .arg(&self.config.model.device)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        for extra in &python_config.extra_args {
            cmd.arg(extra);
        }

        if let Some(cache_dir) = self.config.model.cache_dir.as_ref() {
            cmd.env("CT2_CACHE_DIR", cache_dir);
        }

        for (key, value) in &python_config.env {
            cmd.env(key, value);
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| KlarnetError::Stt(format!("Failed to start Whisper process: {}", e)))?;

        self.process = Some(child);

        info!("Whisper process started");
        Ok(())
    }

    pub async fn transcribe_batch(&self, pcm: &[f32]) -> KlarnetResult<Transcript> {
        // Simplified implementation - in production, use actual faster-whisper
        let segments = self.process_audio(pcm).await?;

        let full_text = segments
            .iter()
            .map(|s| s.text.clone())
            .collect::<Vec<_>>()
            .join(" ");

        Ok(Transcript {
            id: Uuid::new_v4(),
            language: self.config.language.clone(),
            segments,
            full_text,
            processing_time: Duration::from_millis(50),
        })
    }

    async fn process_audio(&self, pcm: &[f32]) -> KlarnetResult<Vec<TranscriptSegment>> {
        // Mock implementation for demonstration
        // In production, send PCM data to faster-whisper and parse response

        let duration = pcm.len() as f64 / 16000.0;

        // Simulate word-level timestamps
        let words = vec![
            WordInfo {
                word: "джарвис".to_string(),
                start: 0.0,
                end: 0.5,
                confidence: 0.95,
            },
            WordInfo {
                word: "включи".to_string(),
                start: 0.5,
                end: 1.0,
                confidence: 0.92,
            },
            WordInfo {
                word: "свет".to_string(),
                start: 1.0,
                end: 1.3,
                confidence: 0.94,
            },
        ];

        Ok(vec![TranscriptSegment {
            start: 0.0,
            end: duration,
            text: "джарвис включи свет".to_string(),
            confidence: 0.93,
            words,
        }])
    }
}
