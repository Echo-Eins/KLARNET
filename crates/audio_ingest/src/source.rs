use std::sync::Arc;
#[cfg(feature = "hardware")]
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use klarnet_core::{AudioConfig, AudioFrame, KlarnetError, KlarnetResult};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};
use tracing::info;

#[cfg(feature = "hardware")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(feature = "hardware")]
use cpal::{Device, Stream, StreamConfig};
#[cfg(feature = "hardware")]
use tracing::{error, warn};

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
    stream: Mutex<Option<Stream>>,
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
            stream: Mutex::new(None),
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
        *self
            .stream
            .lock()
            .map_err(|_| KlarnetError::Audio("Failed to lock audio stream".to_string()))? =
            Some(stream);
        Ok(())
    }

    async fn stop(&mut self) -> KlarnetResult<()> {
        if let Some(stream) = self
            .stream
            .lock()
            .map_err(|_| KlarnetError::Audio("Failed to lock audio stream".to_string()))?
            .take()
        {
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
#[allow(dead_code)]
pub struct MicrophoneSource;

#[cfg(not(feature = "hardware"))]
#[allow(dead_code)]
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

pub struct StubSource {
    task: Option<JoinHandle<()>>,
}

impl StubSource {
    pub fn new() -> Self {
        Self { task: None }
    }
}

#[async_trait]
impl AudioSource for StubSource {
    async fn start(
        &mut self,
        tx: mpsc::UnboundedSender<AudioFrame>,
        config: AudioConfig,
    ) -> KlarnetResult<()> {
        if self.task.is_some() {
            return Err(KlarnetError::Audio(
                "Stub audio source already running".to_string(),
            ));
        }

        let frame_duration = if config.sample_rate > 0 {
            Duration::from_secs_f32(config.buffer_size as f32 / config.sample_rate as f32)
        } else {
            Duration::from_millis(0)
        };
        let channels = config.channels as usize;
        let frame_len = config.buffer_size * channels;
        let sample_rate = config.sample_rate;
        let mut ticker = interval(frame_duration.max(Duration::from_millis(1)));
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

        let task = tokio::spawn(async move {
            loop {
                ticker.tick().await;
                let frame = AudioFrame {
                    data: Arc::from(vec![0.0; frame_len].into_boxed_slice()),
                    timestamp: Utc::now(),
                    duration: frame_duration,
                    sample_rate,
                };

                if tx.send(frame).is_err() {
                    break;
                }
            }
        });

        self.task = Some(task);
        info!("Stub audio source started");
        Ok(())
    }

    async fn stop(&mut self) -> KlarnetResult<()> {
        if let Some(task) = self.task.take() {
            task.abort();
            info!("Stub audio source stopped");
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "Stub"
    }
}