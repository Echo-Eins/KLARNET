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
use cpal::{Device, Sample, SampleFormat, Stream, StreamConfig, SupportedStreamConfig};
#[cfg(feature = "hardware")]
use tracing::{error, warn};
use dasp_sample::conv::ToSample;
#[cfg(feature = "hardware")]
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
    selected_name: String,
}

#[cfg(feature = "hardware")]
unsafe impl Send for MicrophoneSource {}

#[cfg(feature = "hardware")]
unsafe impl Sync for MicrophoneSource {}

#[cfg(feature = "hardware")]
impl MicrophoneSource {
    pub fn new(preferred: Option<&str>) -> KlarnetResult<Self> {
        let host = cpal::default_host();
        let device = preferred
            .and_then(|name| find_input_device(&host, name))
            .or_else(|| host.default_input_device());

        let Some(device) = device else {
            return Err(KlarnetError::Audio("No input device available".to_string()));
        };

        let name = device
            .name()
            .unwrap_or_else(|_| "Unnamed input device".to_string());

        if let Some(target) = preferred {
            info!(
                "Selected audio input device '{}' (requested '{}')",
                name, target
            );
        } else {
            info!("Using default audio input device '{}'", name);
        }

        Ok(Self {
            device: Some(device),
            stream: Mutex::new(None),
            selected_name: name,
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

        let (mut stream_config, sample_format) = Self::negotiate_stream_config(device, &config)?;

        // `negotiate_stream_config` always sets the buffer size to the device default, but
        // calling `build_input_stream` requires a mutable reference. Reaffirm here so that
        // future changes don't forget this requirement.
        stream_config.buffer_size = cpal::BufferSize::Default;

        let actual_sample_rate = stream_config.sample_rate.0;
        let actual_channels = stream_config.channels;

        if actual_sample_rate != config.sample_rate || actual_channels != config.channels {
            info!(
                requested_sample_rate = config.sample_rate,
                requested_channels = config.channels,
                actual_sample_rate,
                actual_channels,
                "Adjusted microphone stream configuration to match device capabilities"
            );
        }

        let err_fn = |err| error!("Audio stream error: {}", err);

        let stream = match sample_format {
            SampleFormat::F32 => {
                let tx = tx.clone();
                device
                    .build_input_stream(
                        &stream_config,
                        move |data: &[f32], _| {
                            Self::handle_input_frame(
                                &tx,
                                data,
                                actual_sample_rate,
                                actual_channels,
                            );
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| KlarnetError::Audio(e.to_string()))?
            }
            SampleFormat::I16 => {
                let tx = tx.clone();
                device
                    .build_input_stream(
                        &stream_config,
                        move |data: &[i16], _| {
                            Self::handle_input_frame(
                                &tx,
                                data,
                                actual_sample_rate,
                                actual_channels,
                            );
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| KlarnetError::Audio(e.to_string()))?
            }
            SampleFormat::U8 => {
                let tx = tx.clone();
                device
                    .build_input_stream(
                        &stream_config,
                        move |data: &[u8], _| {
                            // Конвертируем U8 → F32
                            let converted: Vec<f32> = data
                                .iter()
                                .map(|&sample| {
                                    // U8: 0-255 → F32: -1.0 to 1.0
                                    (sample as f32 / 128.0) - 1.0
                                })
                                .collect();

                            Self::handle_input_frame(
                                &tx,
                                &converted,
                                actual_sample_rate,
                                actual_channels,
                            );
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| KlarnetError::Audio(e.to_string()))?
            }
            SampleFormat::U16 => {
                let tx = tx.clone();
                device
                    .build_input_stream(
                        &stream_config,
                        move |data: &[u16], _| {
                            Self::handle_input_frame(
                                &tx,
                                data,
                                actual_sample_rate,
                                actual_channels,
                            );
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|e| KlarnetError::Audio(e.to_string()))?
            }
            other => {
                return Err(KlarnetError::Audio(format!(
                    "Unsupported input sample format reported by device: {other:?}"
                )))
            }
        };

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
        &self.selected_name
    }
}

#[cfg(feature = "hardware")]
impl MicrophoneSource {
    fn negotiate_stream_config(
        device: &Device,
        requested: &AudioConfig,
    ) -> KlarnetResult<(StreamConfig, SampleFormat)> {
        let mut selected: Option<SupportedStreamConfig> = None;

        match device.supported_input_configs() {
            Ok(configs) => {
                for range in configs {
                    if range.channels() == requested.channels {
                        if range.min_sample_rate().0 <= requested.sample_rate
                            && range.max_sample_rate().0 >= requested.sample_rate
                        {
                            selected = Some(
                                range.with_sample_rate(cpal::SampleRate(requested.sample_rate)),
                            );
                            break;
                        }

                        let fallback_rate = requested
                            .sample_rate
                            .clamp(range.min_sample_rate().0, range.max_sample_rate().0);
                        selected = Some(range.with_sample_rate(cpal::SampleRate(fallback_rate)));
                    } else if selected.is_none() {
                        selected = Some(range.with_sample_rate(range.max_sample_rate()));
                    }
                }
            }
            Err(err) => {
                warn!("Failed to query supported input configurations: {err}");
            }
        }

        if selected.is_none() {
            if let Ok(default_config) = device.default_input_config() {
                selected = Some(default_config);
            }
        }

        let Some(selected) = selected else {
            return Err(KlarnetError::Audio(
                "No supported input stream configurations reported by device".to_string(),
            ));
        };

        let mut config = selected.config();
        config.buffer_size = cpal::BufferSize::Default;

        Ok((config, selected.sample_format()))
    }

    fn handle_input_frame<T>(
        tx: &mpsc::UnboundedSender<AudioFrame>,
        data: &[T],
        sample_rate: u32,
        channels: u16,
    ) where
        T: Sample + ToSample<f32>,
    {
        if data.is_empty() || sample_rate == 0 || channels == 0 {
            return;
        }

        let pcm: Vec<f32> = data
            .iter()
            .copied()
            .map(|sample| sample.to_sample::<f32>())
            .collect();
        let channels_usize = channels as usize;
        let duration = Duration::from_secs_f32(
            pcm.len() as f32 / (sample_rate as f32 * channels_usize as f32),
        );

        let frame = AudioFrame {
            data: Arc::from(pcm.into_boxed_slice()),
            timestamp: Utc::now(),
            duration,
            sample_rate,
            channels,
        };

        if let Err(err) = tx.send(frame) {
            warn!("Failed to send audio frame: {err}");
        }
    }
}


#[cfg(feature = "hardware")]
fn find_input_device(host: &cpal::Host, name: &str) -> Option<Device> {
    let target = name.trim().to_ascii_lowercase();
    if target.is_empty() {
        return None;
    }

    if let Ok(devices) = host.input_devices() {
        for device in devices {
            if let Ok(current_name) = device.name() {
                if current_name.to_ascii_lowercase() == target {
                    return Some(device);
                }
            }
        }
    }
    None
}

#[cfg(not(feature = "hardware"))]
#[allow(dead_code)]
pub struct MicrophoneSource;

#[cfg(not(feature = "hardware"))]
#[allow(dead_code)]
impl MicrophoneSource {
    pub fn new(_preferred: Option<&str>) -> KlarnetResult<Self> {
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
                    channels: config.channels,
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
