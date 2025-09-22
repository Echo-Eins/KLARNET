// crates/audio_ingest/src/lib.rs

use async_trait::async_trait;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};
use klarnet_core::{AudioConfig, AudioFrame, KlarnetError, KlarnetResult};
use parking_lot::RwLock;
use ringbuf::{HeapConsumer, HeapProducer, HeapRb};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

pub mod source;
pub mod processor;
pub mod resampler;

use source::AudioSource;
use processor::AudioProcessor;

/// Audio capture manager
pub struct AudioIngest {
    config: AudioConfig,
    source: Box<dyn AudioSource>,
    processor: AudioProcessor,
    tx: mpsc::UnboundedSender<AudioFrame>,
    rx: Option<mpsc::UnboundedReceiver<AudioFrame>>,
    pre_roll_buffer: Arc<RwLock<Vec<f32>>>,
    pre_roll_duration: Duration,
}

impl AudioIngest {
    pub fn new(config: AudioConfig, pre_roll_duration: Duration) -> KlarnetResult<Self> {
        let (tx, rx) = mpsc::unbounded_channel();
        let processor = AudioProcessor::new(config.clone());
        let source = Box::new(MicrophoneSource::new()?);

        let pre_roll_samples = ((pre_roll_duration.as_secs_f32() * config.sample_rate as f32) as usize)
            .min(config.sample_rate as usize * 5); // Max 5 seconds pre-roll

        Ok(Self {
            config,
            source,
            processor,
            tx,
            rx: Some(rx),
            pre_roll_buffer: Arc::new(RwLock::new(Vec::with_capacity(pre_roll_samples))),
            pre_roll_duration,
        })
    }

    pub async fn start(&mut self) -> KlarnetResult<mpsc::UnboundedReceiver<AudioFrame>> {
        info!("Starting audio ingest");

        let rx = self.rx.take()
            .ok_or_else(|| KlarnetError::Audio("Audio ingest already started".to_string()))?;

        self.source.start(self.tx.clone(), self.config.clone()).await?;

        info!("Audio ingest started successfully");
        Ok(rx)
    }

    pub async fn stop(&mut self) -> KlarnetResult<()> {
        info!("Stopping audio ingest");
        self.source.stop().await?;
        Ok(())
    }

    pub fn get_pre_roll(&self) -> Vec<f32> {
        self.pre_roll_buffer.read().clone()
    }
}