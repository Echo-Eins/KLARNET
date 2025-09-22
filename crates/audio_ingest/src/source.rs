use async_trait::async_trait;
use klarnet_core::{AudioConfig, AudioFrame, KlarnetError, KlarnetResult};
use tokio::sync::mpsc;

#[cfg(feature = "hardware")]
use std::sync::Arc;
#[cfg(feature = "hardware")]
use std::time::Duration;
#[cfg(feature = "hardware")]
use tracing::{error, info, warn};

#[cfg(feature = "hardware")]
use chrono::Utc;
#[cfg(feature = "hardware")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(feature = "hardware")]
use cpal::{Device, Stream, StreamConfig};

#[async_trait]
pub trait AudioSource: Send + Sync {
    async fn start(
        &mut self,
        tx: mpsc::UnboundedSender<AudioFrame>,
        config: AudioConfig,
    ) -> KlarnetResult<()>;
    async fn stop(&mut self) -> KlarnetResult<()>;
    fn name(&self) -> &str;
}

#[cfg(feature = "hardware")]
pub struct MicrophoneSource {
    device: Option<Device>,
    stream: Option<Stream>,
}

#[cfg(feature = "hardware")]
impl MicrophoneSource {
    pub fn new() -> KlarnetResult<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| KlarnetError::Audio("No input device available".to_string()))?;

        info!("Using audio device: {}", device.name().unwrap_or_default());

        Ok(Self {
            device: Some(device),
            stream: None,
        })
    }

    fn create_stream(
        &mut self,
        tx: mpsc::UnboundedSender<AudioFrame>,
        config: AudioConfig,
    ) -> KlarnetResult<Stream> {
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| KlarnetError::Audio("Device not initialized".to_string()))?;

        let cpal_config = StreamConfig {
            channels: config.channels,
            sample_rate: cpal::SampleRate(config.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(config.buffer_size as u32),
        };

        let err_fn = |err| error!("Audio stream error: {}", err);

        let stream = device
            .build_input_stream(
                &cpal_config,
                move |data: &[f32], _: &_| {
                    let frame = AudioFrame {
                        data: Arc::from(data.to_vec().into_boxed_slice()),
                        timestamp: Utc::now(),
                        duration: Duration::from_secs_f32(
                            data.len() as f32 / config.sample_rate as f32,
                        ),
                        sample_rate: config.sample_rate,
                    };

                    if let Err(e) = tx.send(frame) {
                        warn!("Failed to send audio frame: {}", e);
                    }
                },
                err_fn,
                None,
            )
            .map_err(|e| KlarnetError::Audio(e.to_string()))?;

        Ok(stream)
    }
}

#[cfg(feature = "hardware")]
#[async_trait]
impl AudioSource for MicrophoneSource {
    async fn start(
        &mut self,
        tx: mpsc::UnboundedSender<AudioFrame>,
        config: AudioConfig,
    ) -> KlarnetResult<()> {
        let stream = self.create_stream(tx, config)?;
        stream
            .play()
            .map_err(|e| KlarnetError::Audio(e.to_string()))?;
        self.stream = Some(stream);
        Ok(())
    }

    async fn stop(&mut self) -> KlarnetResult<()> {
        if let Some(stream) = self.stream.take() {
            stream
                .pause()
                .map_err(|e| KlarnetError::Audio(e.to_string()))?;
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "Microphone"
    }
}

#[cfg(not(feature = "hardware"))]
pub struct MicrophoneSource;

#[cfg(not(feature = "hardware"))]
impl MicrophoneSource {
    pub fn new() -> KlarnetResult<Self> {
        Err(KlarnetError::Audio(
            "hardware audio capture not available in this build".to_string(),
        ))
    }
}

#[cfg(not(feature = "hardware"))]
#[async_trait]
impl AudioSource for MicrophoneSource {
    async fn start(
        &mut self,
        _tx: mpsc::UnboundedSender<AudioFrame>,
        _config: AudioConfig,
    ) -> KlarnetResult<()> {
        Err(KlarnetError::Audio(
            "hardware audio capture not available in this build".to_string(),
        ))
    }

    async fn stop(&mut self) -> KlarnetResult<()> {
        Ok(())
    }

    fn name(&self) -> &str {
        "Unavailable"
    }
}
