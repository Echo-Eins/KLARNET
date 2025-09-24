// crates/tts/src/player.rs

use std::time::Duration;

use klarnet_core::{KlarnetError, KlarnetResult};
#[cfg(feature = "hardware-audio")]
use tracing::warn;
use tracing::{debug, info};

#[cfg(feature = "hardware-audio")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(feature = "hardware-audio")]
use cpal::{SampleFormat, StreamConfig};
#[cfg(feature = "hardware-audio")]
use std::collections::VecDeque;
#[cfg(feature = "hardware-audio")]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "hardware-audio")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "hardware-audio")]
use tokio::sync::Notify;

pub struct AudioPlayer {
    backend: AudioBackend,
}

enum AudioBackend {
    #[cfg(feature = "hardware-audio")]
    Real(RealAudioPlayer),
    Simulated(SimulatedAudioPlayer),
}

impl AudioPlayer {
    pub fn new(preferred_device: Option<&str>) -> KlarnetResult<Self> {
        #[cfg(feature = "hardware-audio")]
        {
            match RealAudioPlayer::new(preferred_device) {
                Ok(real) => {
                    return Ok(Self {
                        backend: AudioBackend::Real(real),
                    });
                }
                Err(err) => {
                    warn!("Falling back to simulated audio playback: {err}");
                }
            }
        }

        #[cfg(not(feature = "hardware-audio"))]
        let _ = preferred_device;

        let simulated = SimulatedAudioPlayer::new();
        Ok(Self {
            backend: AudioBackend::Simulated(simulated),
        })
    }

    pub async fn play(&self, audio_data: &[u8], sample_rate: u32) -> KlarnetResult<()> {
        self.play_with_format(audio_data, sample_rate, 1).await
    }

    pub async fn play_with_format(
        &self,
        audio_data: &[u8],
        sample_rate: u32,
        channels: u16,
    ) -> KlarnetResult<()> {
        if audio_data.is_empty() {
            return Ok(());
        }

        if sample_rate == 0 {
            return Err(KlarnetError::Audio(
                "Invalid sample rate configured for playback".to_string(),
            ));
        }

        if channels == 0 {
            return Err(KlarnetError::Audio(
                "PCM payload reports zero channels".to_string(),
            ));
        }

        if audio_data.len() % 2 != 0 {
            return Err(KlarnetError::Audio(
                "PCM payload must contain 16-bit little-endian samples".to_string(),
            ));
        }

        let samples: Vec<i16> = audio_data
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        match &self.backend {
            #[cfg(feature = "hardware-audio")]
            AudioBackend::Real(real) => {
                let prepared = real.prepare_samples(&samples, channels, sample_rate)?;
                let report = real.play(prepared).await?;
                debug!(
                    device = %real.device_name,
                    duration_ms = report.duration.as_millis(),
                    rms = report.rms,
                    underruns = report.underruns,
                    overruns = report.overruns,
                    "Completed audio playback via CPAL"
                );
            }
            AudioBackend::Simulated(sim) => {
                let report = sim.play(&samples, sample_rate, channels).await;
                debug!(
                    duration_ms = report.duration.as_millis(),
                    rms = report.rms,
                    "Simulated audio playback"
                );
            }
        }

        Ok(())
    }
}

struct SimulatedAudioPlayer;

impl SimulatedAudioPlayer {
    fn new() -> Self {
        info!("Audio player running in simulation mode; no audio will be emitted");
        Self
    }

    async fn play(&self, samples: &[i16], sample_rate: u32, channels: u16) -> PlaybackReport {
        let total_samples = samples.len() as u64;
        let channels = channels.max(1) as u64;
        let duration = if sample_rate == 0 {
            Duration::from_millis(0)
        } else {
            Duration::from_secs_f64(total_samples as f64 / (sample_rate as f64 * channels as f64))
        };

        let rms = if total_samples == 0 {
            0.0
        } else {
            let sum_sq: f64 = samples
                .iter()
                .map(|sample| {
                    let normalized = *sample as f64 / i16::MAX as f64;
                    normalized * normalized
                })
                .sum();
            (sum_sq / total_samples as f64).sqrt() as f32
        };

        tokio::time::sleep(duration).await;

        PlaybackReport {
            duration,
            rms,
            underruns: 0,
            overruns: 0,
            total_samples,
        }
    }
}

#[cfg_attr(not(feature = "hardware-audio"), allow(dead_code))]
#[derive(Default)]
struct PlaybackReport {
    duration: Duration,
    rms: f32,
    underruns: u64,
    overruns: u64,
    total_samples: u64,
}

#[cfg(feature = "hardware-audio")]
const BUFFER_CAPACITY_SOFT_LIMIT: usize = 48_000 * 10 * 2; // roughly ten seconds of stereo audio

#[cfg(feature = "hardware-audio")]
struct RealAudioPlayer {
    device_name: String,
    stream: cpal::Stream,
    config: StreamConfig,
    sample_format: SampleFormat,
    state: Arc<PlayerState>,
    buffer_soft_limit: usize,
}

#[cfg(feature = "hardware-audio")]
impl RealAudioPlayer {
    fn new(preferred_device: Option<&str>) -> KlarnetResult<Self> {
        let hosts = cpal::available_hosts();
        if hosts.is_empty() {
            return Err(KlarnetError::Audio(
                "No CPAL hosts available on this system".to_string(),
            ));
        }

        let preferred_lower = preferred_device.map(|name| name.to_ascii_lowercase());
        let mut selected_device: Option<(cpal::Device, String)> = None;

        for host_id in &hosts {
            let host = cpal::host_from_id(*host_id).map_err(|err| {
                KlarnetError::Audio(format!(
                    "Failed to initialize audio host {host_id:?}: {err}"
                ))
            })?;

            let host_name = format!("{host_id:?}");
            if let Ok(devices) = host.output_devices() {
                for device in devices {
                    let name = device
                        .name()
                        .unwrap_or_else(|_| "Unnamed output device".to_string());
                    info!(host = %host_name, device = %name, "Detected audio output device");
                    if let Some(preferred) = &preferred_lower {
                        if name.to_ascii_lowercase().contains(preferred) {
                            selected_device = Some((device, name));
                            break;
                        }
                    }
                }
            }

            if selected_device.is_some() {
                break;
            }
        }

        if selected_device.is_none() {
            let default_host = cpal::default_host();
            if let Some(default_device) = default_host.default_output_device() {
                let name = default_device
                    .name()
                    .unwrap_or_else(|_| "Default output device".to_string());
                info!(device = %name, "Using default audio output device");
                selected_device = Some((default_device, name));
            }
        }

        let (device, device_name) = selected_device.ok_or_else(|| {
            KlarnetError::Audio("No audio output device could be selected".to_string())
        })?;

        let supported_config = device
            .default_output_config()
            .map_err(|err| KlarnetError::Audio(format!("Failed to query device config: {err}")))?;
        let sample_format = supported_config.sample_format();
        let config = supported_config.config();

        info!(
            device = %device_name,
            sample_rate = config.sample_rate.0,
            channels = config.channels,
            format = ?sample_format,
            "Starting CPAL audio stream"
        );

        let state = Arc::new(PlayerState::new());
        let buffer_soft_limit = BUFFER_CAPACITY_SOFT_LIMIT
            .max(config.sample_rate.0 as usize * config.channels as usize * 2);

        let stream_state = state.clone();
        let channels = config.channels as usize;

        let err_fn = move |err| {
            warn!("Audio stream error: {err}");
        };

        let stream = match sample_format {
            SampleFormat::I16 => {
                let state = stream_state.clone();
                device
                    .build_output_stream(
                        &config,
                        move |data: &mut [i16], _| {
                            render_output(data, &state, convert_i16);
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|err| {
                        KlarnetError::Audio(format!("Failed to build i16 stream: {err}"))
                    })?
            }
            SampleFormat::F32 => {
                let state = stream_state.clone();
                device
                    .build_output_stream(
                        &config,
                        move |data: &mut [f32], _| {
                            render_output(data, &state, convert_f32);
                        },
                        err_fn,
                        None,
                    )
                    .map_err(|err| {
                        KlarnetError::Audio(format!("Failed to build f32 stream: {err}"))
                    })?
            }
            SampleFormat::U16 => device
                .build_output_stream(
                    &config,
                    move |data: &mut [u16], _| {
                        render_output(data, &stream_state, convert_u16);
                    },
                    err_fn,
                    None,
                )
                .map_err(|err| KlarnetError::Audio(format!("Failed to build u16 stream: {err}")))?,
            other => {
                return Err(KlarnetError::Audio(format!(
                    "Unsupported sample format reported by device: {other:?}"
                )))
            }
        };

        stream
            .play()
            .map_err(|err| KlarnetError::Audio(format!("Failed to start audio stream: {err}")))?;

        Ok(Self {
            device_name,
            stream,
            config,
            sample_format,
            state,
            buffer_soft_limit,
        })
    }

    fn prepare_samples(
        &self,
        input: &[i16],
        input_channels: u16,
        input_sample_rate: u32,
    ) -> KlarnetResult<Vec<i16>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        let input_channels = input_channels as usize;
        if input.len() % input_channels != 0 {
            return Err(KlarnetError::Audio(
                "PCM frame count does not align with reported channel count".to_string(),
            ));
        }

        let frames = input.len() / input_channels;
        let mut channel_data = vec![Vec::with_capacity(frames); input_channels];
        for (index, sample) in input.iter().enumerate() {
            let channel = index % input_channels;
            channel_data[channel].push(*sample as f32 / i16::MAX as f32);
        }

        let resampled_channels: Vec<Vec<f32>> = if input_sample_rate == self.config.sample_rate.0 {
            channel_data
        } else {
            channel_data
                .into_iter()
                .map(|channel| {
                    resample_linear(&channel, input_sample_rate, self.config.sample_rate.0)
                })
                .collect()
        };

        if resampled_channels.is_empty() {
            return Ok(Vec::new());
        }

        let output_channels = self.config.channels as usize;
        let frames = resampled_channels
            .iter()
            .map(|channel| channel.len())
            .min()
            .unwrap_or(0);

        if frames == 0 {
            return Ok(Vec::new());
        }

        let mut interleaved = Vec::with_capacity(frames * output_channels);

        if output_channels == 1 {
            for frame_idx in 0..frames {
                let mut sum = 0.0f32;
                for channel in &resampled_channels {
                    sum += channel[frame_idx];
                }
                interleaved.push(clamp_to_i16(sum / resampled_channels.len() as f32));
            }
        } else if resampled_channels.len() == 1 {
            let channel = &resampled_channels[0];
            for frame_idx in 0..frames {
                let sample = clamp_to_i16(channel[frame_idx]);
                for _ in 0..output_channels {
                    interleaved.push(sample);
                }
            }
        } else {
            for frame_idx in 0..frames {
                for channel_idx in 0..output_channels {
                    let source_idx = channel_idx % resampled_channels.len();
                    let sample = clamp_to_i16(resampled_channels[source_idx][frame_idx]);
                    interleaved.push(sample);
                }
            }
        }

        Ok(interleaved)
    }

    async fn play(&self, samples: Vec<i16>) -> KlarnetResult<PlaybackReport> {
        if samples.is_empty() {
            return Ok(PlaybackReport::default());
        }

        let before = self.state.metrics_snapshot();
        self.enqueue_samples(samples)?;

        loop {
            let notified = self.state.notify.notified();
            if self.state.pending_samples.load(Ordering::SeqCst) == 0 {
                break;
            }
            notified.await;
            if self.state.pending_samples.load(Ordering::SeqCst) == 0 {
                break;
            }
        }

        let after = self.state.metrics_snapshot();
        Ok(after.delta(&before, self.config.sample_rate.0, self.config.channels))
    }

    fn enqueue_samples(&self, samples: Vec<i16>) -> KlarnetResult<()> {
        if samples.is_empty() {
            return Ok(());
        }

        let overflow;
        {
            let mut buffer = self
                .state
                .buffer
                .lock()
                .expect("player buffer mutex poisoned");
            let new_len = buffer.len() + samples.len();
            overflow = new_len.saturating_sub(self.buffer_soft_limit);
            buffer.extend(samples.into_iter());
        }

        if overflow > 0 {
            let mut metrics = self.state.metrics.lock().expect("metrics mutex poisoned");
            metrics.overruns += overflow as u64;
        }

        self.state
            .pending_samples
            .fetch_add(samples.len(), Ordering::SeqCst);

        Ok(())
    }
}

#[cfg(feature = "hardware-audio")]
struct PlayerState {
    buffer: Mutex<VecDeque<i16>>,
    metrics: Mutex<PlaybackMetrics>,
    pending_samples: AtomicUsize,
    notify: Notify,
}

#[cfg(feature = "hardware-audio")]
impl PlayerState {
    fn new() -> Self {
        Self {
            buffer: Mutex::new(VecDeque::new()),
            metrics: Mutex::new(PlaybackMetrics::default()),
            pending_samples: AtomicUsize::new(0),
            notify: Notify::new(),
        }
    }

    fn metrics_snapshot(&self) -> PlaybackMetrics {
        self.metrics.lock().expect("metrics mutex poisoned").clone()
    }
}

#[cfg(feature = "hardware-audio")]
#[derive(Default, Clone)]
struct PlaybackMetrics {
    total_samples: u64,
    sum_squares: f64,
    underruns: u64,
    overruns: u64,
}

#[cfg(feature = "hardware-audio")]
impl PlaybackMetrics {
    fn delta(&self, previous: &PlaybackMetrics, sample_rate: u32, channels: u16) -> PlaybackReport {
        let samples_delta = self.total_samples.saturating_sub(previous.total_samples);
        let sum_squares_delta = self.sum_squares - previous.sum_squares;
        let underruns = self.underruns.saturating_sub(previous.underruns);
        let overruns = self.overruns.saturating_sub(previous.overruns);

        let duration = if sample_rate == 0 || channels == 0 {
            Duration::from_millis(0)
        } else {
            Duration::from_secs_f64(samples_delta as f64 / (sample_rate as f64 * channels as f64))
        };

        let rms = if samples_delta == 0 {
            0.0
        } else {
            let mean = (sum_squares_delta / samples_delta as f64).max(0.0);
            mean.sqrt() as f32
        };

        PlaybackReport {
            duration,
            rms,
            underruns,
            overruns,
            total_samples: samples_delta as u64,
        }
    }
}

#[cfg(feature = "hardware-audio")]
fn render_output<T, F>(data: &mut [T], state: &Arc<PlayerState>, convert: F)
where
    F: Fn(i16) -> (T, f32),
{
    let mut consumed = 0usize;
    let mut processed = 0u64;
    let mut sum_squares = 0f64;
    let mut underruns = 0u64;

    {
        let mut buffer = state.buffer.lock().expect("player buffer mutex poisoned");
        for sample in data.iter_mut() {
            if let Some(value) = buffer.pop_front() {
                consumed += 1;
                processed += 1;
                let (converted, normalized) = convert(value);
                sum_squares += (normalized as f64) * (normalized as f64);
                *sample = converted;
            } else {
                processed += 1;
                let (converted, _) = convert(0);
                *sample = converted;
                underruns += 1;
            }
        }
    }

    if consumed > 0 {
        let previous = state.pending_samples.fetch_sub(consumed, Ordering::SeqCst);
        if previous <= consumed {
            state.notify.notify_waiters();
        }
    }

    if processed > 0 {
        let mut metrics = state.metrics.lock().expect("metrics mutex poisoned");
        metrics.total_samples += processed;
        metrics.sum_squares += sum_squares;
        metrics.underruns += underruns;
    }
}

#[cfg(feature = "hardware-audio")]
fn convert_i16(sample: i16) -> (i16, f32) {
    let normalized = sample as f32 / i16::MAX as f32;
    (sample, normalized)
}

#[cfg(feature = "hardware-audio")]
fn convert_f32(sample: i16) -> (f32, f32) {
    let normalized = sample as f32 / i16::MAX as f32;
    (normalized, normalized)
}

#[cfg(feature = "hardware-audio")]
fn convert_u16(sample: i16) -> (u16, f32) {
    let normalized = sample as f32 / i16::MAX as f32;
    let shifted = (sample as i32 + i16::MAX as i32 + 1).clamp(0, u16::MAX as i32) as u16;
    (shifted, normalized)
}

#[cfg(feature = "hardware-audio")]
fn resample_linear(channel: &[f32], input_rate: u32, output_rate: u32) -> Vec<f32> {
    if channel.is_empty() || input_rate == 0 || output_rate == 0 {
        return Vec::new();
    }

    if input_rate == output_rate {
        return channel.to_vec();
    }

    let output_len = ((channel.len() as f64) * output_rate as f64 / input_rate as f64)
        .round()
        .max(1.0) as usize;

    if channel.len() == 1 {
        return vec![channel[0]; output_len];
    }

    let ratio = input_rate as f64 / output_rate as f64;
    let mut output = Vec::with_capacity(output_len);

    for index in 0..output_len {
        let position = index as f64 * ratio;
        let base = position.floor() as usize;
        let next = (base + 1).min(channel.len() - 1);
        let fraction = (position - base as f64) as f32;
        let interpolated = channel[base] * (1.0 - fraction) + channel[next] * fraction;
        output.push(interpolated);
    }

    output
}

#[cfg(feature = "hardware-audio")]
fn clamp_to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

#[cfg(all(test, feature = "hardware-audio"))]
mod hardware_tests {
    use super::*;

    #[test]
    fn linear_resampler_handles_empty() {
        assert!(resample_linear(&[], 48_000, 24_000).is_empty());
    }

    #[test]
    fn linear_resampler_changes_length() {
        let input = vec![0.0f32, 0.5, -0.5, 0.25];
        let output = resample_linear(&input, 16_000, 48_000);
        assert!(output.len() > input.len());
    }

    #[test]
    fn clamp_to_i16_limits_range() {
        assert_eq!(clamp_to_i16(2.0), i16::MAX);
        assert_eq!(clamp_to_i16(-2.0), i16::MIN);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn simulated_player_reports_metrics() {
        let player = SimulatedAudioPlayer::new();
        let samples = vec![i16::MAX, 0, -i16::MAX];
        let report = player.play(&samples, 48_000, 1).await;
        assert!(report.duration > Duration::from_millis(0));
        assert!(report.rms > 0.0);
    }
}