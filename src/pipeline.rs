use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use audio_ingest::AudioIngest;
use chrono::{DateTime, Utc};
use klarnet_core::{
    AudioChunk, AudioConfig, AudioFrame, CommandType, NluResult, Transcript, TranscriptSegment,
    VadEvent, WordInfo,
};
use nlu::{NluConfig, NluEngine};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::task::JoinHandle;
use tokio::time;
use tracing::{debug, error, info, warn};
use uuid::Uuid;
use vad::{VadConfig, VadProcessor};
use whisper_stt::{WhisperConfig, WhisperEngine};

const MAX_SEGMENT_DURATION_MS: u64 = 12_000;
const INITIAL_BACKOFF_MS: u64 = 200;
const MAX_BACKOFF_MS: u64 = 5_000;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimulationConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub scripted_transcripts: Vec<String>,
    #[serde(default = "default_simulation_interval_ms")]
    pub interval_ms: u64,
}

fn default_simulation_interval_ms() -> u64 {
    250
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    #[serde(default = "default_wake_word")]
    pub wake_word: String,
    #[serde(default = "default_pre_roll_ms")]
    pub pre_roll_ms: u64,
    #[serde(default)]
    pub vad: VadConfig,
    #[serde(default)]
    pub stt: WhisperConfig,
    #[serde(default)]
    pub nlu: NluConfig,
    #[serde(default)]
    pub simulation: SimulationConfig,
}

fn default_wake_word() -> String {
    "Кларнет".to_string()
}

fn default_pre_roll_ms() -> u64 {
    500
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            wake_word: default_wake_word(),
            pre_roll_ms: default_pre_roll_ms(),
            vad: VadConfig::default(),
            stt: WhisperConfig::default(),
            nlu: NluConfig::default(),
            simulation: SimulationConfig::default(),
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
    transcripts: usize,
    commands: usize,
    last_activity: Option<Instant>,
    last_chunk_latency: Option<Duration>,
    max_chunk_latency: Option<Duration>,
    segment_overflows: usize,
}

impl PipelineMetrics {
    pub fn processed_frames(&self) -> usize {
        self.processed_frames
    }

    pub fn generated_chunks(&self) -> usize {
        self.generated_chunks
    }

    pub fn processed_transcripts(&self) -> usize {
        self.transcripts
    }

    pub fn emitted_commands(&self) -> usize {
        self.commands
    }

    pub fn last_activity(&self) -> Option<Instant> {
        self.last_activity
    }

    pub fn last_chunk_latency(&self) -> Option<Duration> {
        self.last_chunk_latency
    }

    pub fn max_chunk_latency(&self) -> Option<Duration> {
        self.max_chunk_latency
    }

    pub fn segment_overflows(&self) -> usize {
        self.segment_overflows
    }
}

pub struct AudioPipeline {
    config: PipelineConfig,
    audio_config: AudioConfig,
    control_tx: Option<mpsc::Sender<ControlMessage>>,
    task: Option<JoinHandle<()>>,
    metrics: Arc<Mutex<PipelineMetrics>>,
    stt_rx: Option<mpsc::UnboundedReceiver<Transcript>>,
    nlu_rx: Option<mpsc::UnboundedReceiver<NluResult>>,
}

impl AudioPipeline {
    pub fn new(config: PipelineConfig, audio_config: AudioConfig) -> Self {
        Self {
            config,
            audio_config,
            control_tx: None,
            task: None,
            metrics: Arc::new(Mutex::new(PipelineMetrics::default())),
            stt_rx: None,
            nlu_rx: None,
        }
    }

    pub async fn start(&mut self) -> Result<(), PipelineError> {
        if self.task.is_some() {
            return Err(PipelineError::AlreadyRunning);
        }

        let (control_tx, control_rx) = mpsc::channel(1);
        let (stt_tx, stt_rx) = mpsc::unbounded_channel();
        let (nlu_tx, nlu_rx) = mpsc::unbounded_channel();
        let metrics = Arc::clone(&self.metrics);
        let config = self.config.clone();
        let audio_config = self.audio_config.clone();

        let task = tokio::spawn(async move {
            if config.simulation.enabled {
                run_simulated_pipeline(control_rx, config, metrics, stt_tx, nlu_tx).await;
            } else {
                run_realtime_pipeline(control_rx, config, audio_config, metrics, stt_tx, nlu_tx)
                    .await;
            }
        });

        self.control_tx = Some(control_tx);
        self.task = Some(task);
        self.stt_rx = Some(stt_rx);
        self.nlu_rx = Some(nlu_rx);

        info!("Audio pipeline started");
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
            transcripts = metrics.processed_transcripts(),
            commands = metrics.emitted_commands(),
            segment_overflows = metrics.segment_overflows(),
            last_chunk_latency_ms = metrics
                .last_chunk_latency()
                .map(|d| d.as_millis() as u64)
                .unwrap_or_default(),
            max_chunk_latency_ms = metrics
                .max_chunk_latency()
                .map(|d| d.as_millis() as u64)
                .unwrap_or_default(),
            last_inactive_ms = last_active_ms,
            "Audio pipeline stopped"
        );

        Ok(())
    }

    pub async fn metrics(&self) -> PipelineMetrics {
        self.metrics.lock().await.clone()
    }

    pub fn take_stt_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<Transcript>> {
        self.stt_rx.take()
    }

    pub fn take_nlu_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<NluResult>> {
        self.nlu_rx.take()
    }
}

async fn run_realtime_pipeline(
    mut control_rx: mpsc::Receiver<ControlMessage>,
    config: PipelineConfig,
    audio_config: AudioConfig,
    metrics: Arc<Mutex<PipelineMetrics>>,
    stt_tx: mpsc::UnboundedSender<Transcript>,
    nlu_tx: mpsc::UnboundedSender<NluResult>,
) {
    info!("Starting realtime audio pipeline");

    let mut backoff = Duration::from_millis(INITIAL_BACKOFF_MS);

    loop {
        match control_rx.try_recv() {
            Ok(ControlMessage::Shutdown { ack }) => {
                info!("Shutdown requested before realtime pipeline initialisation completed");
                let _ = ack.send(());
                break;
            }
            Err(TryRecvError::Disconnected) => {
                info!("Control channel disconnected before realtime pipeline start");
                break;
            }
            Err(TryRecvError::Empty) => {}
        }

        match PipelineComponents::initialise(&config, &audio_config).await {
            Ok(mut components) => {
                backoff = Duration::from_millis(INITIAL_BACKOFF_MS);

                match run_pipeline_session(
                    &mut control_rx,
                    &mut components,
                    &metrics,
                    &stt_tx,
                    &nlu_tx,
                )
                    .await
                {
                    SessionOutcome::Shutdown => {
                        components.shutdown().await;
                        break;
                    }
                    SessionOutcome::Restart => {
                        warn!("Realtime pipeline session requested restart");
                        components.shutdown().await;
                        backoff = (backoff * 2).min(Duration::from_millis(MAX_BACKOFF_MS));
                        time::sleep(backoff).await;
                    }
                }
            }
            Err(err) => {
                error!(error = %err, "Failed to initialise realtime pipeline components");
                backoff = (backoff * 2).min(Duration::from_millis(MAX_BACKOFF_MS));
                tokio::select! {
                    _ = time::sleep(backoff) => {},
                    Some(ControlMessage::Shutdown { ack }) = control_rx.recv() => {
                        info!("Shutdown requested while waiting to retry component initialisation");
                        let _ = ack.send(());
                        break;
                    }
                }
            }
        }
    }
    info!("Realtime audio pipeline terminated");
}

struct CompletedChunk {
    chunk: AudioChunk,
    overflowed: bool,
}

struct SegmentCollector {
    collecting: bool,
    pre_roll_frames: VecDeque<AudioFrame>,
    pre_roll_total: Duration,
    max_pre_roll: Duration,
    active_frames: Vec<AudioFrame>,
    active_duration: Duration,
    max_chunk_duration: Duration,
    sample_rate: u32,
}

impl SegmentCollector {
    fn new(sample_rate: u32, pre_roll_ms: u64, max_chunk_ms: u64) -> Self {
        let pre_roll_duration = Duration::from_millis(pre_roll_ms);
        let mut effective_max = max_chunk_ms.max(pre_roll_ms);
        if effective_max == 0 {
            effective_max = 1000;
        }
        let max_chunk_duration = Duration::from_millis(effective_max);

        let frame_capacity = ((sample_rate as u64 * pre_roll_ms) / 1000).max(1) as usize;
        let pre_roll_frames = VecDeque::with_capacity(frame_capacity);

        Self {
            collecting: false,
            pre_roll_frames,
            pre_roll_total: Duration::from_millis(0),
            max_pre_roll: pre_roll_duration,
            active_frames: Vec::new(),
            active_duration: Duration::from_millis(0),
            max_chunk_duration,
            sample_rate,
        }
    }

    fn observe_frame(&mut self, frame: &AudioFrame) {
        if self.collecting {
            return;
        }

        self.pre_roll_frames.push_back(frame.clone());
        self.pre_roll_total += frame.duration;

        while self.pre_roll_total > self.max_pre_roll {
            if let Some(oldest) = self.pre_roll_frames.pop_front() {
                self.pre_roll_total = self.pre_roll_total.saturating_sub(oldest.duration);
            } else {
                self.pre_roll_total = Duration::from_millis(0);
                break;
            }
        }
    }

    fn start(&mut self) {
        if self.collecting {
            return;
        }
        self.collecting = true;
        self.active_frames.clear();
        self.active_duration = Duration::from_millis(0);

        for frame in self.pre_roll_frames.drain(..) {
            self.active_duration += frame.duration;
            self.active_frames.push(frame);
        }

        self.pre_roll_total = Duration::from_millis(0);
    }

    fn push_speech_frame(&mut self, frame: AudioFrame) -> Vec<CompletedChunk> {
        if !self.collecting {
            self.start();
        }

        self.active_duration += frame.duration;
        self.active_frames.push(frame);

        if self.active_duration >= self.max_chunk_duration {
            vec![self.complete_current(true)]
        } else {
            Vec::new()
        }
    }

    fn push_pcm_frame(&mut self, timestamp: DateTime<Utc>, pcm: Vec<f32>) -> Vec<CompletedChunk> {
        let frame = self.make_frame(timestamp, pcm);
        self.push_speech_frame(frame)
    }

    fn finish(&mut self) -> Option<CompletedChunk> {
        if !self.collecting {
            return None;
        }

        self.collecting = false;
        if self.active_frames.is_empty() {
            self.reset();
            return None;
        }

        let chunk = self.complete_current(false);
        self.reset();
        Some(chunk)
    }

    fn on_silence(&mut self) {
        if self.collecting {
            return;
        }
        self.pre_roll_frames.clear();
        self.pre_roll_total = Duration::from_millis(0);
    }

    fn complete_current(&mut self, overflowed: bool) -> CompletedChunk {
        let frames = std::mem::take(&mut self.active_frames);
        self.active_duration = Duration::from_millis(0);

        CompletedChunk {
            chunk: AudioChunk::new(frames),
            overflowed,
        }
    }

    fn reset(&mut self) {
        self.collecting = false;
        self.active_frames.clear();
        self.active_duration = Duration::from_millis(0);
        self.pre_roll_frames.clear();
        self.pre_roll_total = Duration::from_millis(0);
    }

    fn make_frame(&self, timestamp: DateTime<Utc>, pcm: Vec<f32>) -> AudioFrame {
        let duration = if pcm.is_empty() {
            Duration::from_millis(0)
        } else {
            Duration::from_secs_f32(pcm.len() as f32 / self.sample_rate as f32)
        };

        AudioFrame {
            data: Arc::from(pcm.into_boxed_slice()),
            timestamp,
            duration,
            sample_rate: self.sample_rate,
        }
    }
}

struct PipelineComponents {
    audio_ingest: AudioIngest,
    audio_rx: mpsc::UnboundedReceiver<AudioFrame>,
    vad_processor: VadProcessor,
    vad_rx: mpsc::UnboundedReceiver<VadEvent>,
    stt_engine: WhisperEngine,
    nlu_engine: NluEngine,
    collector: SegmentCollector,
}

impl PipelineComponents {
    async fn initialise(
        config: &PipelineConfig,
        audio_config: &AudioConfig,
    ) -> Result<Self, String> {
        let mut audio_ingest = AudioIngest::new(
            audio_config.clone(),
            Duration::from_millis(config.pre_roll_ms),
        )
            .map_err(|err| format!("audio ingest init failed: {err}"))?;

        let audio_rx = audio_ingest
            .start()
            .await
            .map_err(|err| format!("audio ingest start failed: {err}"))?;

        let (vad_processor, vad_rx) = VadProcessor::new(config.vad.clone())
            .map_err(|err| format!("vad init failed: {err}"))?;

        let stt_engine = WhisperEngine::new(config.stt.clone())
            .await
            .map_err(|err| format!("stt init failed: {err}"))?;

        let mut nlu_config = config.nlu.clone();
        if !nlu_config
            .wake_words
            .iter()
            .any(|word| word.eq_ignore_ascii_case(&config.wake_word))
        {
            nlu_config.wake_words.push(config.wake_word.clone());
        }

        let nlu_engine = NluEngine::new(nlu_config)
            .await
            .map_err(|err| format!("nlu init failed: {err}"))?;

        if let Some(summary) = nlu_engine.llm_configuration() {
            info!(
                provider = %summary.provider,
                model = %summary.model,
                cache_enabled = summary.cache_enabled,
                max_concurrent_requests = summary.max_concurrent_requests,
                min_request_interval_ms = summary.min_request_interval_ms,
                "Configured NLU LLM backend",
            );
        }

        let collector = SegmentCollector::new(
            audio_config.sample_rate,
            config.pre_roll_ms,
            MAX_SEGMENT_DURATION_MS,
        );

        Ok(Self {
            audio_ingest,
            audio_rx,
            vad_processor,
            vad_rx,
            stt_engine,
            nlu_engine,
            collector,
        })
    }

    async fn shutdown(&mut self) {
        if let Err(err) = self.audio_ingest.stop().await {
            warn!("Failed to stop audio ingest: {err}");
        }
        self.stt_engine.shutdown();
        self.collector.reset();
    }
}

enum SessionOutcome {
    Shutdown,
    Restart,
}

async fn run_pipeline_session(
    control_rx: &mut mpsc::Receiver<ControlMessage>,
    components: &mut PipelineComponents,
    metrics: &Arc<Mutex<PipelineMetrics>>,
    stt_tx: &mpsc::UnboundedSender<Transcript>,
    nlu_tx: &mpsc::UnboundedSender<NluResult>,
) -> SessionOutcome {

    loop {
        tokio::select! {
            control = control_rx.recv() => {
                match control {
                    Some(ControlMessage::Shutdown { ack }) => {
                        info!("Shutting down realtime audio pipeline session");
                        let _ = ack.send(());
                        return SessionOutcome::Shutdown;
                    }
                    None => {
                        info!("Control channel closed, stopping realtime pipeline session");
                        return SessionOutcome::Shutdown;
                    }
                }
            }
            Some(frame) = components.audio_rx.recv() => {
                {
                    let mut guard = metrics.lock().await;
                    guard.processed_frames += 1;
                    guard.last_activity = Some(Instant::now());
                }

                components.collector.observe_frame(&frame);

                if let Err(err) = components.vad_processor.process_frame(frame).await {
                    warn!("VAD processing error, restarting session: {err}");
                    return SessionOutcome::Restart;
                }
            }
            Some(event) = components.vad_rx.recv() => {
                let mut chunks = Vec::new();
                match event {
                    VadEvent::SpeechStart { timestamp, confidence } => {
                        debug!(?timestamp, confidence, "Speech start detected");
                        components.collector.start();
                    }
                    VadEvent::SpeechFrame { timestamp, pcm, .. } => {
                        chunks.extend(components.collector.push_pcm_frame(timestamp, pcm));
                    }
                    VadEvent::SpeechEnd { timestamp, duration } => {
                        debug!(?timestamp, ?duration, "Speech end detected");
                        if let Some(chunk) = components.collector.finish() {
                            chunks.push(chunk);
                        }
                    }
                    VadEvent::Silence { timestamp, duration } => {
                        debug!(?timestamp, ?duration, "Silence window detected");
                        components.collector.on_silence();
                    }
                }

                for completed in chunks {
                    if let Err(err) = process_audio_chunk(
                        completed,
                        metrics,
                        &mut components.stt_engine,
                        &components.nlu_engine,
                        stt_tx,
                        nlu_tx,
                    )
                    .await
                    {
                        warn!("Failed to process audio chunk, restarting session: {err}");
                        return SessionOutcome::Restart;
                    }
                }
            }
            else => {
                warn!("Realtime pipeline channel closed unexpectedly, restarting session");
                return SessionOutcome::Restart;
            }
        }
    }
}

async fn process_audio_chunk(
    completed: CompletedChunk,
    metrics: &Arc<Mutex<PipelineMetrics>>,
    stt_engine: &mut WhisperEngine,
    nlu_engine: &NluEngine,
    stt_tx: &mpsc::UnboundedSender<Transcript>,
    nlu_tx: &mpsc::UnboundedSender<NluResult>,
) -> Result<(), String> {
    let CompletedChunk { chunk, overflowed } = completed;
    if chunk.frames.is_empty() {
        return Ok(());
    }

    let chunk_duration = chunk
        .frames
        .iter()
        .fold(Duration::from_millis(0), |acc, frame| acc + frame.duration);

    let latency = chrono_duration_to_std(Utc::now().signed_duration_since(chunk.end_time));

    {
        let mut guard = metrics.lock().await;
        guard.generated_chunks += 1;
        guard.last_activity = Some(Instant::now());
        guard.last_chunk_latency = Some(latency);
        if guard
            .max_chunk_latency
            .map(|max| latency > max)
            .unwrap_or(true)
        {
            guard.max_chunk_latency = Some(latency);
        }
        if overflowed {
            guard.segment_overflows += 1;
        }
    }

    info!(
        chunk_id = %chunk.id,
        frames = chunk.frames.len(),
        duration_ms = chunk_duration.as_millis() as u64,
        latency_ms = latency.as_millis() as u64,
        overflowed,
        "Submitting audio chunk to STT"
    );

    match stt_engine.transcribe(chunk).await {
        Ok(transcript) => {
            {
                let mut guard = metrics.lock().await;
                guard.transcripts += 1;
                guard.last_activity = Some(Instant::now());
            }

            if stt_tx.send(transcript.clone()).is_err() {
                warn!("STT consumer dropped transcript channel");
            }
            dispatch_nlu(transcript, metrics, nlu_engine, nlu_tx).await;
            Ok(())
        }
        Err(err) => {
            error!("STT transcription failed: {err}");
            Err(err.to_string())
        }
    }
}

fn chrono_duration_to_std(duration: chrono::Duration) -> Duration {
    duration
        .to_std()
        .unwrap_or_else(|_| Duration::from_millis(0))
}

async fn dispatch_nlu(
    transcript: Transcript,
    metrics: &Arc<Mutex<PipelineMetrics>>,
    nlu_engine: &NluEngine,
    nlu_tx: &mpsc::UnboundedSender<NluResult>,
) {
    match nlu_engine.process(&transcript).await {
        Ok(result) => {
            {
                let mut guard = metrics.lock().await;
                if matches!(result.command_type, CommandType::Local(_)) {
                    guard.commands += 1;
                }
                guard.last_activity = Some(Instant::now());
            }

            if let Some(usage) = nlu_engine.take_last_llm_usage().await {
                info!(
                    provider = %usage.provider,
                    model = %usage.model,
                    prompt_tokens = usage.usage.prompt_tokens,
                    completion_tokens = usage.usage.completion_tokens,
                    total_tokens = usage.usage.total_tokens,
                    latency_ms = usage.latency.as_millis() as u64,
                    "LLM usage recorded",
                );

                if let Some(snapshot) = nlu_engine.llm_metrics_snapshot() {
                    debug!(
                        provider = %snapshot.provider,
                        total_requests = snapshot.total_requests,
                        successful_requests = snapshot.successful_requests,
                        failed_requests = snapshot.failed_requests,
                        cache_hits = snapshot.cache_hits,
                        total_tokens = snapshot.total_tokens_used,
                        avg_latency_ms = snapshot.average_response_time_ms,
                        "LLM metrics snapshot",
                    );
                }
            }

            if let Err(err) = nlu_tx.send(result) {
                warn!("Failed to send NLU result: {err}");
            }
        }
        Err(err) => {
            error!("NLU processing failed: {err}");
        }
    }
}

async fn run_simulated_pipeline(
    mut control_rx: mpsc::Receiver<ControlMessage>,
    config: PipelineConfig,
    metrics: Arc<Mutex<PipelineMetrics>>,
    stt_tx: mpsc::UnboundedSender<Transcript>,
    nlu_tx: mpsc::UnboundedSender<NluResult>,
) {
    info!("Starting simulated pipeline mode");

    let mut nlu_config = config.nlu.clone();
    if !nlu_config
        .wake_words
        .iter()
        .any(|word| word.eq_ignore_ascii_case(&config.wake_word))
    {
        nlu_config.wake_words.push(config.wake_word.clone());
    }

    let nlu_engine = match NluEngine::new(nlu_config).await {
        Ok(engine) => engine,
        Err(err) => {
            error!("Failed to initialise NLU engine: {err}");
            wait_for_shutdown(control_rx).await;
            return;
        }
    };

    let interval = Duration::from_millis(config.simulation.interval_ms);
    let mut scripted = config
        .simulation
        .scripted_transcripts
        .clone()
        .into_iter()
        .peekable();

    while let Some(text) = scripted.next() {
        process_scripted_text(&text, &config, &metrics, &stt_tx, &nlu_engine, &nlu_tx).await;

        if scripted.peek().is_some() {
            tokio::select! {
                _ = time::sleep(interval) => {}
                Some(control) = control_rx.recv() => {
                    match control {
                        ControlMessage::Shutdown { ack } => {
                            let _ = ack.send(());
                            return;
                        }
                    }
                }
            }
        }
    }

    // Wait for shutdown signal to finish gracefully
    wait_for_shutdown(control_rx).await;
}

async fn process_scripted_text(
    text: &str,
    config: &PipelineConfig,
    metrics: &Arc<Mutex<PipelineMetrics>>,
    stt_tx: &mpsc::UnboundedSender<Transcript>,
    nlu_engine: &NluEngine,
    nlu_tx: &mpsc::UnboundedSender<NluResult>,
) {
    let words: Vec<&str> = text.split_whitespace().collect();
    let segments = vec![TranscriptSegment {
        start: 0.0,
        end: words.len() as f64 * 0.5,
        text: text.to_string(),
        confidence: 1.0,
        words: words
            .iter()
            .enumerate()
            .map(|(idx, word)| WordInfo {
                word: (*word).to_string(),
                start: idx as f64 * 0.5,
                end: (idx + 1) as f64 * 0.5,
                confidence: 1.0,
            })
            .collect(),
    }];

    let transcript = Transcript {
        id: Uuid::new_v4(),
        language: config.stt.language.clone(),
        segments,
        full_text: text.to_string(),
        processing_time: Duration::from_millis(10),
    };

    {
        let mut guard = metrics.lock().await;
        guard.generated_chunks += 1;
        guard.transcripts += 1;
        guard.last_activity = Some(Instant::now());
        let zero = Duration::from_millis(0);
        guard.last_chunk_latency = Some(zero);
        if guard.max_chunk_latency.is_none() {
            guard.max_chunk_latency = Some(zero);
        }
    }

    let _ = stt_tx.send(transcript.clone());
    dispatch_nlu(transcript, metrics, nlu_engine, nlu_tx).await;
}

async fn wait_for_shutdown(mut control_rx: mpsc::Receiver<ControlMessage>) {
    if let Some(ControlMessage::Shutdown { ack }) = control_rx.recv().await {
        let _ = ack.send(());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::timeout;
    use chrono::Utc;
    use std::sync::Arc;

    #[tokio::test]
    async fn simulated_pipeline_emits_results() {
        let mut config = PipelineConfig::default();
        config.simulation.enabled = true;
        config.simulation.scripted_transcripts = vec!["Кларнет включи свет".to_string()];
        config.simulation.interval_ms = 10;

        let mut pipeline = AudioPipeline::new(config, AudioConfig::default());
        pipeline.start().await.expect("pipeline starts");

        let mut nlu_rx = pipeline
            .take_nlu_receiver()
            .expect("nlu receiver available");
        let result = timeout(Duration::from_secs(1), nlu_rx.recv())
            .await
            .expect("result ready")
            .expect("nlu result");

        assert!(result.wake_word_detected);

        pipeline.stop().await.expect("pipeline stops");
    }
    fn build_audio_frame(duration_ms: u64, sample_rate: u32) -> AudioFrame {
        let samples = ((duration_ms as f32 / 1000.0) * sample_rate as f32) as usize;
        let pcm = vec![0.0; samples];
        AudioFrame {
            data: Arc::from(pcm.into_boxed_slice()),
            timestamp: Utc::now(),
            duration: Duration::from_millis(duration_ms),
            sample_rate,
        }
    }

    #[test]
    fn segment_collector_preserves_pre_roll() {
        let sample_rate = 16_000;
        let mut collector = SegmentCollector::new(sample_rate, 200, 1_000);

        let pre_roll_frame = build_audio_frame(20, sample_rate);
        collector.observe_frame(&pre_roll_frame);
        collector.start();

        let speech_ts = Utc::now();
        let speech_chunks = collector.push_pcm_frame(speech_ts, vec![1.0; 320]);
        assert!(speech_chunks.is_empty());

        let finished = collector.finish().expect("chunk produced");
        assert_eq!(finished.chunk.frames.len(), 2);
        assert!(!finished.overflowed);
    }

    #[test]
    fn segment_collector_splits_on_overflow() {
        let sample_rate = 16_000;
        let mut collector = SegmentCollector::new(sample_rate, 0, 30);
        collector.start();

        let ts = Utc::now();
        assert!(collector.push_pcm_frame(ts, vec![0.2; 320]).is_empty());

        let overflow = collector.push_pcm_frame(ts, vec![0.2; 320]);
        assert_eq!(overflow.len(), 1);
        assert!(overflow[0].overflowed);
        assert!(overflow[0].chunk.frames.len() >= 2);

        assert!(collector.finish().is_none());
    }
}
