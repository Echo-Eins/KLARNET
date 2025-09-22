// crates/whisper_stt/src/lib.rs

use async_trait::async_trait;
use klarnet_core::{
    AudioChunk, KlarnetError, KlarnetResult, Transcript, TranscriptSegment, WordInfo,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

pub mod config;
pub mod processor;
pub mod streaming;

pub use config::WhisperConfig;
pub use processor::WhisperProcessor;
pub use streaming::StreamingWhisper;

/// Whisper configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperConfig {
    pub model_path: PathBuf,
    pub model_size: ModelSize,
    pub compute_type: ComputeType,
    pub language: String,
    pub beam_size: usize,
    pub vad_filter: bool,
    pub word_timestamps: bool,
    pub device: Device,
    pub device_index: Option<usize>,
    pub num_workers: usize,
    pub batch_size: usize,
    pub max_segment_length: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSize {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputeType {
    Int8,
    Int8Float16,
    Int16,
    Float16,
    Float32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda,
    Auto,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/faster-whisper-medium"),
            model_size: ModelSize::Medium,
            compute_type: ComputeType::Int8Float16,
            language: "ru".to_string(),
            beam_size: 5,
            vad_filter: true,
            word_timestamps: true,
            device: Device::Auto,
            device_index: None,
            num_workers: 1,
            batch_size: 1,
            max_segment_length: 30.0,
        }
    }
}

/// Whisper STT engine
pub struct WhisperEngine {
    config: WhisperConfig,
    processor: Arc<WhisperProcessor>,
    streaming: Option<StreamingWhisper>,
    metrics: Arc<RwLock<WhisperMetrics>>,
}

#[derive(Debug, Default)]
struct WhisperMetrics {
    total_processed: u64,
    total_duration_s: f64,
    avg_processing_time_ms: f64,
    last_processing_time_ms: f64,
    errors: u64,
}

impl WhisperEngine {
    pub async fn new(config: WhisperConfig) -> KlarnetResult<Self> {
        info!("Initializing Whisper engine with model: {:?}", config.model_size);

        let processor = Arc::new(WhisperProcessor::new(config.clone()).await?);
        let streaming = if config.device != Device::Cpu {
            Some(StreamingWhisper::new(config.clone()).await?)
        } else {
            None
        };

        let metrics = Arc::new(RwLock::new(WhisperMetrics::default()));

        Ok(Self {
            config,
            processor,
            streaming,
            metrics,
        })
    }

    pub async fn transcribe(&self, chunk: AudioChunk) -> KlarnetResult<Transcript> {
        let start = Instant::now();
        let pcm_data = chunk.to_pcm();

        debug!(
            "Processing audio chunk: {} samples, {:.2}s",
            pcm_data.len(),
            pcm_data.len() as f32 / 16000.0
        );

        let result = if let Some(streaming) = &self.streaming {
            streaming.transcribe_stream(&pcm_data).await?
        } else {
            self.processor.transcribe_batch(&pcm_data).await?
        };

        let processing_time = start.elapsed();
        self.update_metrics(pcm_data.len(), processing_time);

        Ok(result)
    }

    fn update_metrics(&self, samples: usize, processing_time: Duration) {
        let mut metrics = self.metrics.write();
        metrics.total_processed += 1;
        metrics.total_duration_s += samples as f64 / 16000.0;
        metrics.last_processing_time_ms = processing_time.as_millis() as f64;

        let total_time = metrics.avg_processing_time_ms * (metrics.total_processed - 1) as f64;
        metrics.avg_processing_time_ms =
            (total_time + metrics.last_processing_time_ms) / metrics.total_processed as f64;

        debug!(
            "STT metrics: RTF={:.3}, avg_time={:.1}ms",
            processing_time.as_secs_f64() / (samples as f64 / 16000.0),
            metrics.avg_processing_time_ms
        );
    }

    pub fn get_metrics(&self) -> WhisperMetrics {
        self.metrics.read().clone()
    }
}
