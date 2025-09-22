use std::time::Duration;

use klarnet_core::{AudioConfig, AudioFrame, KlarnetError, KlarnetResult};
use tokio::sync::mpsc;
use tracing::info;

pub struct AudioIngest {
    config: AudioConfig,
    pre_roll_duration: Duration,
    rx: Option<mpsc::UnboundedReceiver<AudioFrame>>,
}

impl AudioIngest {
    pub fn new(config: AudioConfig, pre_roll_duration: Duration) -> KlarnetResult<Self> {
        let (_tx, rx) = mpsc::unbounded_channel();
        Ok(Self {
            config,
            pre_roll_duration,
            rx: Some(rx),
        })
    }

    pub async fn start(&mut self) -> KlarnetResult<mpsc::UnboundedReceiver<AudioFrame>> {
        info!(
            sample_rate = self.config.sample_rate,
            pre_roll_ms = self.pre_roll_duration.as_millis(),
            "Audio ingest stub started"
        );

        self.rx
            .take()
            .ok_or_else(|| KlarnetError::Audio("Audio ingest already started".to_string()))
    }

    pub async fn stop(&mut self) -> KlarnetResult<()> {
        info!("Audio ingest stub stopped");
        Ok(())
    }

    pub fn get_pre_roll(&self) -> Vec<f32> {
        Vec::new()
    }
}
