// src/pipeline.rs
use std::sync::Arc;
use std::time::{Duration, Instant};

use klarnet_core::{AudioChunk, AudioConfig, AudioFrame};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::task::JoinHandle;
use tokio::time;
use tracing::{debug, info, warn};

/// Configuration for the simulated audio pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Simulated wake word that the pipeline will react to.
    #[serde(default = "default_wake_word")]
    pub wake_word: String,
    /// Interval between generated audio frames in milliseconds.
    #[serde(default = "default_frame_interval_ms")]
    pub frame_interval_ms: u64,
    /// Duration of simulated speech activity in milliseconds.
    #[serde(default = "default_speech_duration_ms")]
    pub speech_duration_ms: u64,
    /// After how many milliseconds of inactivity the pipeline auto stops recording.
    #[serde(default = "default_inactivity_timeout_ms")]
    pub inactivity_timeout_ms: u64,
}

fn default_wake_word() -> String {
    "Кларнет".to_string()
}

fn default_frame_interval_ms() -> u64 {
    50
}

fn default_speech_duration_ms() -> u64 {
    1_000
}

fn default_inactivity_timeout_ms() -> u64 {
    2_000
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            wake_word: default_wake_word(),
            frame_interval_ms: default_frame_interval_ms(),
            speech_duration_ms: default_speech_duration_ms(),
            inactivity_timeout_ms: default_inactivity_timeout_ms(),
        }
    }
}

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("pipeline already running")]
    AlreadyRunning,
    #[error("pipeline not running")]
    NotRunning,
    #[error("failed to send control message: {0}")]
    SendError(String),
    #[error("pipeline task failed: {0}")]
    Join(String),
}

#[derive(Debug)]
enum ControlMessage {
    Shutdown { ack: oneshot::Sender<()> },
}

#[derive(Debug, Default, Clone)]
pub struct PipelineMetrics {
    processed_frames: usize,
    generated_chunks: usize,
    last_activity: Option<Instant>,
}

impl PipelineMetrics {
    pub fn processed_frames(&self) -> usize {
        self.processed_frames
    }

    pub fn generated_chunks(&self) -> usize {
        self.generated_chunks
    }

    pub fn last_activity(&self) -> Option<Instant> {
        self.last_activity
    }
}

/// Simplified audio pipeline used for integration testing and local development.
pub struct AudioPipeline {
    config: PipelineConfig,
    audio_config: AudioConfig,
    control_tx: Option<mpsc::Sender<ControlMessage>>,
    task: Option<JoinHandle<()>>,
    metrics: Arc<Mutex<PipelineMetrics>>,
}
impl AudioPipeline {
    pub fn new(config: PipelineConfig, audio_config: AudioConfig) -> Self {
        Self {
            config,
            audio_config,
            control_tx: None,
            task: None,
            metrics: Arc::new(Mutex::new(PipelineMetrics::default())),
        }
    }

    pub async fn start(&mut self) -> Result<(), PipelineError> {
        if self.task.is_some() {
            return Err(PipelineError::AlreadyRunning);
        }

        let (tx, mut rx) = mpsc::channel(1);
        let metrics = Arc::clone(&self.metrics);
        let audio_config = self.audio_config.clone();
        let config = self.config.clone();

        let task = tokio::spawn(async move {
            let mut ticker = time::interval(Duration::from_millis(config.frame_interval_ms));
            let mut simulated_speech_remaining = Duration::from_millis(config.speech_duration_ms);

            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        let mut metrics_guard = metrics.lock().await;
                        let frame = generate_frame(&audio_config);
                        metrics_guard.processed_frames += 1;
                        metrics_guard.last_activity = Some(Instant::now());

                        if simulated_speech_remaining.is_zero() {
                            // Simulate idle period until timeout expires.
                            if let Some(last_activity) = metrics_guard.last_activity {
                                if last_activity.elapsed()
                                    >= Duration::from_millis(config.inactivity_timeout_ms)
                                {
                                    simulated_speech_remaining =
                                        Duration::from_millis(config.speech_duration_ms);
                                    debug!("Simulated wake word '{}' detected", config.wake_word);
                                }

                            }
                            continue;
                        }

                        simulated_speech_remaining = simulated_speech_remaining
                            .saturating_sub(Duration::from_millis(config.frame_interval_ms));
                        drop(metrics_guard);

                        let chunk = AudioChunk::new(vec![frame]);
                        {
                            let mut metrics_guard = metrics.lock().await;
                            metrics_guard.generated_chunks += 1;
                            metrics_guard.last_activity = Some(Instant::now());
                        }
                        info!("Generated audio chunk {}", chunk.id);
                    }
                    Some(ControlMessage::Shutdown { ack }) = rx.recv() => {
                        info!("Pipeline shutdown request received");
                        let _ = ack.send(());
                        break;
                    }
                }
            }
        });

        self.control_tx = Some(tx);
        self.task = Some(task);
        info!(
            "Audio pipeline started (wake word: '{}')",
            self.config.wake_word
        );

        Ok(())
    }

    pub async fn stop(&mut self) -> Result<(), PipelineError> {
        let tx = self.control_tx.take().ok_or(PipelineError::NotRunning)?;
        let task = self.task.take().ok_or(PipelineError::NotRunning)?;

        let (ack_tx, ack_rx) = oneshot::channel();
        tx.send(ControlMessage::Shutdown { ack: ack_tx })
            .await
            .map_err(|err| PipelineError::SendError(err.to_string()))?;

        if ack_rx.await.is_err() {
            warn!("Pipeline shutdown acknowledgement was dropped");
        }

        task.await
            .map_err(|err| PipelineError::Join(err.to_string()))?;

        let metrics = self.metrics().await;
        let last_active_ms = metrics
            .last_activity()
            .map(|instant| instant.elapsed().as_millis() as u64)
            .unwrap_or_default();
        info!(
            processed_frames = metrics.processed_frames(),
            generated_chunks = metrics.generated_chunks(),
            last_inactive_ms = last_active_ms,
            "Audio pipeline stopped"
        );

        Ok(())
    }
    pub async fn metrics(&self) -> PipelineMetrics {
        self.metrics.lock().await.clone()
    }
}

fn generate_frame(config: &AudioConfig) -> AudioFrame {
    use chrono::Utc;
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let samples = config.buffer_size.max(1);
    let mut data = Vec::with_capacity(samples);
    for _ in 0..samples {
        data.push(rng.gen_range(-0.01..0.01));
    }

    AudioFrame {
        data: Arc::from(data.into_boxed_slice()),
        timestamp: Utc::now(),
        duration: Duration::from_secs_f32(samples as f32 / config.sample_rate as f32),
        sample_rate: config.sample_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn pipeline_produces_metrics() {
        let mut pipeline = AudioPipeline::new(
            PipelineConfig {
                frame_interval_ms: 10,
                speech_duration_ms: 100,
                inactivity_timeout_ms: 50,
                wake_word: "тест".to_string(),
            },
            AudioConfig {
                buffer_size: 16,
                ..AudioConfig::default()
            },
        );

        pipeline.start().await.expect("pipeline starts");
        time::sleep(Duration::from_millis(150)).await;

        let metrics = pipeline.metrics().await;
        assert!(metrics.processed_frames() > 0);

        pipeline.stop().await.expect("pipeline stops");
    }
}