// crates/tts/src/player.rs

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
#[cfg(feature = "hardware-audio")]
use std::time::Instant;

use klarnet_core::{KlarnetError, KlarnetResult};
use klarnet_observability::metrics::MetricType;
use klarnet_observability::MetricsCollector;
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
use std::sync::atomic::{AtomicBool, AtomicUsize};
#[cfg(feature = "hardware-audio")]
use std::sync::RwLock;
#[cfg(feature = "hardware-audio")]
use tokio::sync::Notify;
use tokio::sync::Semaphore;

pub struct AudioPlayer {
    backend: AudioBackend,
    concurrency: Arc<Semaphore>,
    metrics: Option<Arc<MetricsCollector>>,
}

enum AudioBackend {
    #[cfg(feature = "hardware-audio")]
    Real(RealAudioPlayer),
    Simulated(SimulatedAudioPlayer),
}

pub struct AudioMixBuffer<'a> {
    pub data: &'a [u8],
    pub sample_rate: u32,
    pub channels: u16,
}

struct PreparedMixBuffer {
    samples: Vec<i16>,
    channels: u16,
    sample_rate: u32,
}

fn decode_pcm(audio_data: &[u8]) -> KlarnetResult<Vec<i16>> {
    if audio_data.len() % 2 != 0 {
        return Err(KlarnetError::Audio(
            "PCM payload must contain 16-bit little-endian samples".to_string(),
        ));
    }

    Ok(audio_data
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

impl AudioPlayer {
    pub fn new(preferred_device: Option<&str>) -> KlarnetResult<Self> {
        Self::with_metrics(preferred_device, None)
    }

    pub fn with_metrics(
        preferred_device: Option<&str>,
        metrics: Option<Arc<MetricsCollector>>,
    ) -> KlarnetResult<Self> {
        let concurrency = Arc::new(Semaphore::new(1));

        #[cfg(feature = "hardware-audio")]
        {
            match RealAudioPlayer::new(preferred_device) {
                Ok(real) => {
                    return Ok(Self {
                        backend: AudioBackend::Real(real),
                        concurrency,
                        metrics,
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
            concurrency,
            metrics,
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

        let samples = decode_pcm(audio_data)?;

        let _permit = self
            .concurrency
            .acquire()
            .await
            .expect("audio semaphore closed");

        match &self.backend {
            #[cfg(feature = "hardware-audio")]
            AudioBackend::Real(real) => {
                const MAX_ATTEMPTS: usize = 3;
                let mut last_err: Option<KlarnetError> = None;

                for attempt in 1..=MAX_ATTEMPTS {
                    match real.play_pcm(&samples, channels, sample_rate).await {
                        Ok(report) => {
                            debug!(
                                device = %real.device_name(),
                                duration_ms = report.duration.as_millis(),
                                rms = report.rms,
                                underruns = report.underruns,
                                overruns = report.overruns,
                                volume = report.current_volume,
                                attempt,
                                "Completed audio playback via CPAL"
                            );
                            self.record_metrics(&report);
                            return Ok(());
                        }
                        Err(err) => {
                            warn!(
                                attempt,
                                max_attempts = MAX_ATTEMPTS,
                                "Audio playback attempt failed: {err}"
                            );
                            last_err = Some(err);
                            real.schedule_restart();
                        }
                    }
                }

                Err(last_err.unwrap_or_else(|| {
                    KlarnetError::Audio("Audio playback failed after multiple attempts".to_string())
                }))
            }
            AudioBackend::Simulated(sim) => {
                let report = sim.play(&samples, sample_rate, channels).await;
                debug!(
                    duration_ms = report.duration.as_millis(),
                    rms = report.rms,
                    volume = report.current_volume,
                    "Simulated audio playback"
                );
                self.record_metrics(&report);
                Ok(())
            }
        }
    }

    pub async fn play_mix(&self, buffers: &[AudioMixBuffer<'_>]) -> KlarnetResult<()> {
        if buffers.is_empty() {
            return Ok(());
        }

        let mut prepared = Vec::with_capacity(buffers.len());
        for buffer in buffers {
            if buffer.sample_rate == 0 {
                return Err(KlarnetError::Audio(
                    "Invalid sample rate configured for playback".to_string(),
                ));
            }

            if buffer.channels == 0 {
                return Err(KlarnetError::Audio(
                    "PCM payload reports zero channels".to_string(),
                ));
            }

            let samples = decode_pcm(buffer.data)?;
            prepared.push(PreparedMixBuffer {
                samples,
                channels: buffer.channels,
                sample_rate: buffer.sample_rate,
            });
        }

        let _permit = self
            .concurrency
            .acquire()
            .await
            .expect("audio semaphore closed");

        match &self.backend {
            #[cfg(feature = "hardware-audio")]
            AudioBackend::Real(real) => {
                let report = real.play_mix(&prepared).await?;
                debug!(
                    device = %real.device_name(),
                    duration_ms = report.duration.as_millis(),
                    rms = report.rms,
                    underruns = report.underruns,
                    overruns = report.overruns,
                    volume = report.current_volume,
                    "Completed mixed audio playback via CPAL"
                );
                self.record_metrics(&report);
                Ok(())
            }
            AudioBackend::Simulated(sim) => {
                let report = sim.play_mix(&prepared).await?;
                debug!(
                    duration_ms = report.duration.as_millis(),
                    rms = report.rms,
                    volume = report.current_volume,
                    "Simulated mixed audio playback"
                );
                self.record_metrics(&report);
                Ok(())
            }
        }
    }
    pub fn set_volume(&self, volume: f32) {
        match &self.backend {
            #[cfg(feature = "hardware-audio")]
            AudioBackend::Real(real) => real.set_volume(volume),
            AudioBackend::Simulated(sim) => sim.set_volume(volume),
        }
    }

    pub fn volume(&self) -> f32 {
        match &self.backend {
            #[cfg(feature = "hardware-audio")]
            AudioBackend::Real(real) => real.volume(),
            AudioBackend::Simulated(sim) => sim.volume(),
        }
    }

    pub fn metrics_snapshot(&self) -> AudioMetricsSnapshot {
        match &self.backend {
            #[cfg(feature = "hardware-audio")]
            AudioBackend::Real(real) => real.metrics_snapshot(),
            AudioBackend::Simulated(sim) => sim.metrics_snapshot(),
        }
    }

    fn record_metrics(&self, report: &PlaybackReport) {
        if let Some(metrics) = &self.metrics {
            metrics.record(
                MetricType::AudioPlaybackDurationMs,
                report.duration.as_millis() as f64,
            );
            metrics.record(MetricType::AudioPlaybackRms, report.rms as f64);
            metrics.add(MetricType::AudioPlaybackUnderruns, report.underruns as f64);
            metrics.add(MetricType::AudioPlaybackOverruns, report.overruns as f64);
            metrics.record(MetricType::AudioCurrentVolume, report.current_volume as f64);
            metrics.add(
                MetricType::AudioFramesProcessed,
                report.total_samples as f64,
            );
        }
    }
}

pub struct AudioMetricsSnapshot {
    pub total_duration: Duration,
    pub rms: f32,
    pub underruns: u64,
    pub overruns: u64,
    pub total_samples: u64,
    pub current_volume: f32,
}

struct SimulatedAudioPlayer {
    volume: AtomicU32,
    metrics: Mutex<PlaybackMetrics>,
    stream_state: Mutex<SimulatedStreamState>,
}

#[derive(Clone)]
struct SimulatedStreamState {
    sample_rate: u32,
    channels: u16,
}

impl Default for SimulatedStreamState {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            channels: 1,
        }
    }
}

impl SimulatedAudioPlayer {
    fn new() -> Self {
        info!("Audio player running in simulation mode; no audio will be emitted");
        Self {
            volume: AtomicU32::new(f32::to_bits(1.0)),
            metrics: Mutex::new(PlaybackMetrics::default()),
            stream_state: Mutex::new(SimulatedStreamState::default()),
        }
    }

    async fn play(&self, samples: &[i16], sample_rate: u32, channels: u16) -> PlaybackReport {
        let total_samples = samples.len() as u64;
        let channels = channels.max(1) as u64;
        let duration = if sample_rate == 0 {
            Duration::from_millis(0)
        } else {
            Duration::from_secs_f64(total_samples as f64 / (sample_rate as f64 * channels as f64))
        };

        let volume = self.volume();
        let sum_sq: f64 = samples
            .iter()
            .map(|sample| {
                let normalized = *sample as f64 / i16::MAX as f64;
                normalized * normalized
            })
            .sum();
        let scaled_sum_sq = sum_sq * (volume as f64 * volume as f64);

        let rms = if total_samples == 0 {
            0.0
        } else {
            (scaled_sum_sq / total_samples as f64).sqrt() as f32
        };

        tokio::time::sleep(duration).await;

        {
            let mut metrics = self.metrics.lock().expect("metrics mutex poisoned");
            metrics.total_samples += total_samples;
            metrics.sum_squares += scaled_sum_sq;
        }

        {
            let mut state = self
                .stream_state
                .lock()
                .expect("simulated stream state mutex poisoned");
            if sample_rate > 0 {
                state.sample_rate = sample_rate;
            }
            if channels as u16 > 0 {
                state.channels = channels as u16;
            }
        }

        PlaybackReport {
            duration,
            rms,
            underruns: 0,
            overruns: 0,
            total_samples,
            current_volume: volume,
        }
    }

    async fn play_mix(&self, buffers: &[PreparedMixBuffer]) -> KlarnetResult<PlaybackReport> {
        if buffers.is_empty() {
            return Ok(PlaybackReport::default());
        }

        let target_sample_rate = buffers[0].sample_rate;
        let target_channels = buffers[0].channels;

        let mut prepared = Vec::with_capacity(buffers.len());
        for buffer in buffers {
            let resampled = prepare_samples_for_output(
                &buffer.samples,
                buffer.channels,
                buffer.sample_rate,
                target_channels,
                target_sample_rate,
            )?;

            if !resampled.is_empty() {
                prepared.push(resampled);
            }
        }

        let mixed = mix_interleaved(&prepared);
        let report = self.play(&mixed, target_sample_rate, target_channels).await;
        Ok(report)
    }

    fn set_volume(&self, volume: f32) {
        let clamped = volume.clamp(0.0, 2.0);
        self.volume.store(f32::to_bits(clamped), Ordering::SeqCst);
    }

    fn volume(&self) -> f32 {
        f32::from_bits(self.volume.load(Ordering::SeqCst))
    }

    fn metrics_snapshot(&self) -> AudioMetricsSnapshot {
        let metrics = self.metrics.lock().expect("metrics mutex poisoned").clone();
        let stream = self
            .stream_state
            .lock()
            .expect("simulated stream state mutex poisoned")
            .clone();
        let volume = self.volume();
        let report = metrics.snapshot(stream.sample_rate, stream.channels, volume);

        AudioMetricsSnapshot {
            total_duration: report.duration,
            rms: report.rms,
            underruns: report.underruns,
            overruns: report.overruns,
            total_samples: metrics.total_samples,
            current_volume: volume,
        }
    }
}

#[cfg_attr(not(feature = "hardware-audio"), allow(dead_code))]
struct PlaybackReport {
    duration: Duration,
    rms: f32,
    underruns: u64,
    overruns: u64,
    total_samples: u64,
    current_volume: f32,
}

impl Default for PlaybackReport {
    fn default() -> Self {
        Self {
            duration: Duration::from_millis(0),
            rms: 0.0,
            underruns: 0,
            overruns: 0,
            total_samples: 0,
            current_volume: 1.0,
        }
    }
}

#[cfg(feature = "hardware-audio")]
const BUFFER_CAPACITY_SOFT_LIMIT: usize = 48_000 * 10 * 2; // roughly ten seconds of stereo audio
#[cfg(feature = "hardware-audio")]
#[cfg(feature = "hardware-audio")]
struct RealAudioPlayer {
    supervisor: Arc<StreamSupervisor>,
}

#[cfg(feature = "hardware-audio")]
impl RealAudioPlayer {
    fn new(preferred_device: Option<&str>) -> KlarnetResult<Self> {

        let state = Arc::new(PlayerState::new());
        let supervisor = StreamSupervisor::new(preferred_device, state)?;
        Ok(Self { supervisor })
    }

    fn prepare_samples(
        &self,
        input: &[i16],
        input_channels: u16,
        input_sample_rate: u32,
    ) -> KlarnetResult<Vec<i16>> {

        let params = self.supervisor.current_params();

        prepare_samples_for_output(
            input,
            input_channels,
            input_sample_rate,
            params.config.channels,
            params.config.sample_rate.0,
        )
    }

    async fn play_prepared(&self, samples: Vec<i16>) -> KlarnetResult<PlaybackReport> {
        if samples.is_empty() {
            return Ok(PlaybackReport::default());
        }

        self.supervisor.ensure_stream().await?;
        let state = self.supervisor.state();

        let before = state.metrics_snapshot();
        self.enqueue_samples(samples)?;

        loop {
            let notified = state.notify.notified();
            if state.pending_samples.load(Ordering::SeqCst) == 0 {
                break;
            }
            notified.await;
            if state.pending_samples.load(Ordering::SeqCst) == 0 {
                break;
            }
            if state.restart_requested() {
                break;
            }
        }

        if state.restart_requested() {
            return Err(KlarnetError::Audio(
                "Audio stream restart requested during playback".to_string(),
            ));
        }

        let after = state.metrics_snapshot();
        let params = self.supervisor.current_params();
        let volume = state.volume();
        Ok(after.delta(
            &before,
            params.config.sample_rate.0,
            params.config.channels,
            volume,
        ))
    }

    async fn play_mix(&self, buffers: &[PreparedMixBuffer]) -> KlarnetResult<PlaybackReport> {
        if buffers.is_empty() {
            return Ok(PlaybackReport::default());
        }

        self.supervisor.ensure_stream().await?;
        let mut prepared = Vec::with_capacity(buffers.len());
        for buffer in buffers {
            let resampled =
                self.prepare_samples(&buffer.samples, buffer.channels, buffer.sample_rate)?;
            if !resampled.is_empty() {
                prepared.push(resampled);
            }
        }

        if prepared.is_empty() {
            return Ok(PlaybackReport::default());
        }

        let mixed = mix_interleaved(&prepared);
        self.play_prepared(mixed).await
    }

    fn enqueue_samples(&self, samples: Vec<i16>) -> KlarnetResult<()> {
        if samples.is_empty() {
            return Ok(());
        }
        let state = self.supervisor.state();
        let soft_limit = self.supervisor.current_buffer_soft_limit();
        let overflow;
        {
            let mut buffer = state.buffer.lock().expect("player buffer mutex poisoned");
            let new_len = buffer.len() + samples.len();
            overflow = new_len.saturating_sub(soft_limit);
            buffer.extend(samples.into_iter());
        }

        if overflow > 0 {
            let mut metrics = state.metrics.lock().expect("metrics mutex poisoned");
            metrics.overruns += overflow as u64;
        }

        state
            .pending_samples
            .fetch_add(samples.len(), Ordering::SeqCst);

        Ok(())
    }
    async fn play_pcm(
        &self,
        input: &[i16],
        input_channels: u16,
        input_sample_rate: u32,
    ) -> KlarnetResult<PlaybackReport> {
        self.supervisor.ensure_stream().await?;
        let prepared = self.prepare_samples(input, input_channels, input_sample_rate)?;
        self.play_prepared(prepared).await
    }

    fn schedule_restart(&self) {
        self.supervisor.schedule_restart();
    }

    fn device_name(&self) -> String {
        self.supervisor.current_params().device_name
    }

    fn set_volume(&self, volume: f32) {
        self.supervisor.state().set_volume(volume);
    }

    fn volume(&self) -> f32 {
        self.supervisor.state().volume()
    }

    fn metrics_snapshot(&self) -> AudioMetricsSnapshot {
        let params = self.supervisor.current_params();
        let state = self.supervisor.state();
        let metrics = state.metrics_snapshot();
        let volume = state.volume();
        let report = metrics.snapshot(params.config.sample_rate.0, params.config.channels, volume);

        AudioMetricsSnapshot {
            total_duration: report.duration,
            rms: report.rms,
            underruns: report.underruns,
            overruns: report.overruns,
            total_samples: metrics.total_samples,
            current_volume: volume,
        }
    }
}

#[cfg(feature = "hardware-audio")]
struct PlayerState {
    buffer: Mutex<VecDeque<i16>>,
    metrics: Mutex<PlaybackMetrics>,
    pending_samples: AtomicUsize,
    notify: Notify,
    restart_requested: AtomicBool,
    volume: AtomicU32,
}

#[cfg(feature = "hardware-audio")]
impl PlayerState {
    fn new() -> Self {
        Self {
            buffer: Mutex::new(VecDeque::new()),
            metrics: Mutex::new(PlaybackMetrics::default()),
            pending_samples: AtomicUsize::new(0),
            notify: Notify::new(),
            restart_requested: AtomicBool::new(false),
            volume: AtomicU32::new(f32::to_bits(1.0)),
        }
    }

    fn metrics_snapshot(&self) -> PlaybackMetrics {
        self.metrics.lock().expect("metrics mutex poisoned").clone()
    }
    fn request_restart(&self) {
        self.restart_requested.store(true, Ordering::SeqCst);
        self.pending_samples.store(0, Ordering::SeqCst);
        self.notify.notify_waiters();
    }

    fn take_restart_request(&self) -> bool {
        self.restart_requested.swap(false, Ordering::SeqCst)
    }

    fn restart_requested(&self) -> bool {
        self.restart_requested.load(Ordering::SeqCst)
    }

    fn set_volume(&self, volume: f32) {
        let clamped = volume.clamp(0.0, 2.0);
        self.volume.store(f32::to_bits(clamped), Ordering::SeqCst);
    }

    fn volume(&self) -> f32 {
        f32::from_bits(self.volume.load(Ordering::SeqCst))
    }

}

#[cfg(feature = "hardware-audio")]
struct StreamSupervisor {
    state: Arc<PlayerState>,
    preferred_device: Option<String>,
    params: Arc<RwLock<StreamParams>>,
    resources: tokio::sync::Mutex<StreamResources>,
    last_device_check: Mutex<Instant>,
}

#[cfg(feature = "hardware-audio")]
impl StreamSupervisor {
    fn new(preferred_device: Option<&str>, state: Arc<PlayerState>) -> KlarnetResult<Arc<Self>> {
        let (resources, params) = Self::build_stream(preferred_device, &state, &[])?;

        Ok(Arc::new(Self {
            state,
            preferred_device: preferred_device.map(|name| name.to_string()),
            params: Arc::new(RwLock::new(params)),
            resources: tokio::sync::Mutex::new(resources),
            last_device_check: Mutex::new(Instant::now()),
        }))
    }

    fn current_params(&self) -> StreamParams {
        self.params.read().expect("stream params poisoned").clone()
    }

    fn current_buffer_soft_limit(&self) -> usize {
        self.current_params().buffer_soft_limit
    }

    fn schedule_restart(&self) {
        self.state.request_restart();
    }

    fn state(&self) -> &Arc<PlayerState> {
        &self.state
    }

    async fn ensure_stream(&self) -> KlarnetResult<()> {
        if self.state.take_restart_request() {
            if let Err(err) = self.rebuild_stream("restart requested").await {
                self.state.request_restart();
                return Err(err);
            }
            return Ok(());
        }

        if self.device_check_due() {
            if let Err(err) = self.verify_device().await {
                warn!("Audio device health check failed: {err}");
                if let Err(rebuild_err) = self.rebuild_stream("device health check failed").await {
                    self.state.request_restart();
                    return Err(rebuild_err);
                }
            }
        }

        Ok(())
    }

    async fn rebuild_stream(&self, reason: &str) -> KlarnetResult<()> {
        let previous = self.current_params().device_name;
        warn!(reason, previous_device = %previous, "Reinitializing audio output stream");

        let (resources, params) = Self::build_stream(
            self.preferred_device.as_deref(),
            &self.state,
            &[previous.clone()],
        )?;

        {
            let mut guard = self.params.write().expect("stream params poisoned");
            *guard = params.clone();
        }

        {
            let mut guard = self.resources.lock().await;
            *guard = resources;
        }

        *self
            .last_device_check
            .lock()
            .expect("device check mutex poisoned") = Instant::now();

        info!(
            device = %params.device_name,
            sample_rate = params.config.sample_rate.0,
            channels = params.config.channels,
            format = ?params.sample_format,
            "Audio stream restarted"
        );

        Ok(())
    }

    fn device_check_due(&self) -> bool {
        const DEVICE_CHECK_INTERVAL: Duration = Duration::from_secs(30);
        let last = *self
            .last_device_check
            .lock()
            .expect("device check mutex poisoned");
        last.elapsed() >= DEVICE_CHECK_INTERVAL
    }

    async fn verify_device(&self) -> KlarnetResult<()> {
        let guard = self.resources.lock().await;
        guard
            .device
            .default_output_config()
            .map_err(|err| KlarnetError::Audio(format!("Audio device unavailable: {err}")))?;
        drop(guard);

        *self
            .last_device_check
            .lock()
            .expect("device check mutex poisoned") = Instant::now();
        Ok(())
    }

    fn build_stream(
        preferred_device: Option<&str>,
        state: &Arc<PlayerState>,
        priority: &[String],
    ) -> KlarnetResult<(StreamResources, StreamParams)> {
        let mut devices = Self::enumerate_devices()?;
        if devices.is_empty() {
            return Err(KlarnetError::Audio(
                "No audio output device could be selected".to_string(),
            ));
        }

        let preferred_lower = preferred_device.map(|name| name.to_ascii_lowercase());

        devices.sort_by(|(_, a_name), (_, b_name)| {
            let a_priority = Self::device_priority(a_name, &preferred_lower, priority);
            let b_priority = Self::device_priority(b_name, &preferred_lower, priority);
            a_priority.cmp(&b_priority)
        });

        let mut last_err: Option<KlarnetError> = None;

        for (device, name) in devices {
            match Self::create_stream_for_device(device, &name, state) {
                Ok(result) => {
                    return Ok(result);
                }
                Err(err) => {
                    warn!(device = %name, "Failed to initialize audio stream: {err}");
                    last_err = Some(err);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| {
            KlarnetError::Audio("Unable to initialize any audio output device".to_string())
        }))
    }

    fn enumerate_devices() -> KlarnetResult<Vec<(cpal::Device, String)>> {
        let hosts = cpal::available_hosts();
        if hosts.is_empty() {
            return Err(KlarnetError::Audio(
                "No CPAL hosts available on this system".to_string(),
            ));
        }

        let mut devices = Vec::new();
        for host_id in hosts {
            let host = cpal::host_from_id(host_id).map_err(|err| {
                KlarnetError::Audio(format!(
                    "Failed to initialize audio host {host_id:?}: {err}"
                ))
            })?;

            if let Ok(output_devices) = host.output_devices() {
                for device in output_devices {
                    let name = device
                        .name()
                        .unwrap_or_else(|_| "Unnamed output device".to_string());
                    info!(host = ?host_id, device = %name, "Detected audio output device");
                    devices.push((device, name));
                }
            }
        }

        if devices.is_empty() {
            let default_host = cpal::default_host();
            if let Some(default_device) = default_host.default_output_device() {
                let name = default_device
                    .name()
                    .unwrap_or_else(|_| "Default output device".to_string());
                info!(device = %name, "Using default audio output device");
                devices.push((default_device, name));
            }
        }

        Ok(devices)
    }

    fn device_priority(
        name: &str,
        preferred: &Option<String>,
        priority: &[String],
    ) -> (u8, usize, String) {
        if let Some(position) = priority.iter().position(|candidate| candidate == name) {
            return (0, position, name.to_string());
        }

        if let Some(preferred_lower) = preferred {
            if name.to_ascii_lowercase().contains(preferred_lower) {
                return (1, 0, name.to_string());
            }
        }

        (2, 0, name.to_string())
    }

    fn create_stream_for_device(
        device: cpal::Device,
        device_name: &str,
        state: &Arc<PlayerState>,
    ) -> KlarnetResult<(StreamResources, StreamParams)> {
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

        let buffer_soft_limit = BUFFER_CAPACITY_SOFT_LIMIT
            .max(config.sample_rate.0 as usize * config.channels as usize * 2);

        let stream_state = state.clone();
        let error_state = state.clone();
        let name_for_error = device_name.to_string();

        let err_fn = move |err| {
            warn!(device = %name_for_error, "Audio stream error: {err}");
            error_state.request_restart();
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

        let params = StreamParams {
            device_name: device_name.to_string(),
            config: config.clone(),
            sample_format,
            buffer_soft_limit,
        };

        Ok((StreamResources { device, stream }, params))
    }
}

#[cfg(feature = "hardware-audio")]
#[derive(Clone)]
struct StreamParams {
    device_name: String,
    config: StreamConfig,
    sample_format: SampleFormat,
    buffer_soft_limit: usize,
}

#[cfg(feature = "hardware-audio")]
struct StreamResources {
    device: cpal::Device,
    stream: cpal::Stream,
}

#[derive(Default, Clone)]
struct PlaybackMetrics {
    total_samples: u64,
    sum_squares: f64,
    underruns: u64,
    overruns: u64,
}

#[cfg(feature = "hardware-audio")]
impl PlaybackMetrics {
    fn delta(
        &self,
        previous: &PlaybackMetrics,
        sample_rate: u32,
        channels: u16,
        volume: f32,
    ) -> PlaybackReport {

        let samples_delta = self.total_samples.saturating_sub(previous.total_samples);
        let sum_squares_delta = (self.sum_squares - previous.sum_squares).max(0.0);
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
            current_volume: volume,
        }
    }

    fn snapshot(&self, sample_rate: u32, channels: u16, volume: f32) -> PlaybackReport {
        let duration = if sample_rate == 0 || channels == 0 {
            Duration::from_millis(0)
        } else {
            Duration::from_secs_f64(
                self.total_samples as f64 / (sample_rate as f64 * channels as f64),
            )
        };

        let rms = if self.total_samples == 0 {
            0.0
        } else {
            let mean = (self.sum_squares / self.total_samples as f64).max(0.0);
            mean.sqrt() as f32
        };

        PlaybackReport {
            duration,
            rms,
            underruns: self.underruns,
            overruns: self.overruns,
            total_samples: self.total_samples,
            current_volume: volume,
        }
    }
}

#[cfg(feature = "hardware-audio")]
fn render_output<T, F>(data: &mut [T], state: &Arc<PlayerState>, convert: F)
where
    F: Fn(f32) -> (T, f32),
{
    let mut consumed = 0usize;
    let mut processed = 0u64;
    let mut sum_squares = 0f64;
    let mut underruns = 0u64;
    let volume = state.volume();

    {
        let mut buffer = state.buffer.lock().expect("player buffer mutex poisoned");
        for sample in data.iter_mut() {
            let normalized = if let Some(value) = buffer.pop_front() {
                consumed += 1;
                processed += 1;
                value as f32 / i16::MAX as f32
            } else {
                processed += 1;
                underruns += 1;
                0.0
            };

            let scaled = (normalized * volume).clamp(-1.0, 1.0);
            let (converted, output_normalized) = convert(scaled);
            sum_squares += (output_normalized as f64) * (output_normalized as f64);
            *sample = converted;
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
fn convert_i16(sample: f32) -> (i16, f32) {
    let clamped = sample.clamp(-1.0, 1.0);
    (clamp_to_i16(clamped), clamped)
}

#[cfg(feature = "hardware-audio")]
fn convert_f32(sample: f32) -> (f32, f32) {
    let clamped = sample.clamp(-1.0, 1.0);
    (clamped, clamped)
}

#[cfg(feature = "hardware-audio")]
fn convert_u16(sample: f32) -> (u16, f32) {
    let clamped = sample.clamp(-1.0, 1.0);
    let int_sample = clamp_to_i16(clamped);
    let shifted = (int_sample as i32 + i16::MAX as i32 + 1).clamp(0, u16::MAX as i32) as u16;
    (shifted, clamped)
}

fn prepare_samples_for_output(
    input: &[i16],
    input_channels: u16,
    input_sample_rate: u32,
    output_channels: u16,
    output_sample_rate: u32,
) -> KlarnetResult<Vec<i16>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let channels = input_channels.max(1) as usize;
    if input.len() % channels != 0 {
        return Err(KlarnetError::Audio(
            "PCM frame count does not align with reported channel count".to_string(),
        ));
    }

    let frames = input.len() / channels;
    let mut channel_data = vec![Vec::with_capacity(frames); channels];
    for (index, sample) in input.iter().enumerate() {
        let channel = index % channels;
        channel_data[channel].push(*sample as f32 / i16::MAX as f32);
    }

    let resampled_channels: Vec<Vec<f32>> = if input_sample_rate == output_sample_rate {
        channel_data
    } else {
        channel_data
            .into_iter()
            .map(|channel| resample_linear(&channel, input_sample_rate, output_sample_rate))
            .collect()
    };

    if resampled_channels.is_empty() {
        return Ok(Vec::new());
    }

    let output_channels = output_channels.max(1) as usize;
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
            let divisor = resampled_channels.len().max(1) as f32;
            interleaved.push(clamp_to_i16(sum / divisor));
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

fn mix_interleaved(buffers: &[Vec<i16>]) -> Vec<i16> {
    if buffers.is_empty() {
        return Vec::new();
    }

    let max_len = buffers.iter().map(|buffer| buffer.len()).max().unwrap_or(0);
    if max_len == 0 {
        return Vec::new();
    }

    let mut mixed = vec![0.0f32; max_len];
    for buffer in buffers {
        for (index, sample) in buffer.iter().enumerate() {
            mixed[index] += *sample as f32 / i16::MAX as f32;
        }
    }

    mixed.into_iter().map(clamp_to_i16).collect()
}

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
        assert_eq!(report.current_volume, 1.0);
    }

    #[tokio::test]
    async fn simulated_player_mixes_buffers() {
        let player = SimulatedAudioPlayer::new();
        let inputs = vec![
            PreparedMixBuffer {
                samples: vec![i16::MAX / 2; 4],
                channels: 1,
                sample_rate: 48_000,
            },
            PreparedMixBuffer {
                samples: vec![i16::MAX / 2; 4],
                channels: 1,
                sample_rate: 48_000,
            },
        ];

        let report = player.play_mix(&inputs).await.expect("mixing failed");
        assert!(report.rms > 0.0);
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
    #[test]
    fn audio_player_adjusts_volume() {
        let player = AudioPlayer::new(None).expect("player initialization failed");
        player.set_volume(0.5);
        assert!((player.volume() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn mix_interleaved_adds_buffers() {
        let first = vec![i16::MAX / 2; 2];
        let second = vec![i16::MAX / 2; 2];
        let mixed = mix_interleaved(&[first, second]);
        assert_eq!(mixed.len(), 2);
        assert_eq!(mixed[0], i16::MAX);
    }
}