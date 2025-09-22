use std::sync::Arc;
use std::time::{Duration, Instant};

use audio_ingest::AudioIngest;
use klarnet_core::{
    AudioChunk, AudioConfig, AudioFrame, CommandType, NluResult, Transcript, TranscriptSegment,
    VadEvent, WordInfo,
};
use nlu::{NluConfig, NluEngine};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::task::JoinHandle;
use tokio::time;
use tracing::{debug, error, info, warn};
use uuid::Uuid;
use vad::{VadConfig, VadProcessor};
use whisper_stt::{WhisperConfig, WhisperEngine};

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

    let mut audio_ingest = match AudioIngest::new(
        audio_config.clone(),
        Duration::from_millis(config.pre_roll_ms),
    ) {
        Ok(ingest) => ingest,
        Err(err) => {
            error!("Failed to initialise audio ingest: {err}");
            wait_for_shutdown(control_rx).await;
            return;
        }
    };

    let mut audio_rx = match audio_ingest.start().await {
        Ok(rx) => rx,
        Err(err) => {
            error!("Failed to start audio ingest: {err}");
            wait_for_shutdown(control_rx).await;
            return;
        }
    };

    let (mut vad_processor, mut vad_rx) = match VadProcessor::new(config.vad.clone()) {
        Ok(tuple) => tuple,
        Err(err) => {
            error!("Failed to create VAD processor: {err}");
            let _ = audio_ingest.stop().await;
            wait_for_shutdown(control_rx).await;
            return;
        }
    };

    let mut stt_engine = match WhisperEngine::new(config.stt.clone()).await {
        Ok(engine) => engine,
        Err(err) => {
            error!("Failed to initialise STT engine: {err}");
            let _ = audio_ingest.stop().await;
            wait_for_shutdown(control_rx).await;
            return;
        }
    };

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
            let _ = audio_ingest.stop().await;
            wait_for_shutdown(control_rx).await;
            return;
        }
    };

    let mut collecting = false;
    let mut collected_frames: Vec<AudioFrame> = Vec::new();
    let sample_rate = audio_config.sample_rate;

    loop {
        tokio::select! {
            Some(control) = control_rx.recv() => {
                match control {
                    ControlMessage::Shutdown { ack } => {
                        info!("Shutting down realtime audio pipeline");
                        if let Err(err) = audio_ingest.stop().await {
                            warn!("Failed to stop audio ingest: {err}");
                        }
                        let _ = ack.send(());
                        break;
                    }
                }
            }
            Some(frame) = audio_rx.recv() => {
                {
                    let mut guard = metrics.lock().await;
                    guard.processed_frames += 1;
                    guard.last_activity = Some(Instant::now());
                }

                if let Err(err) = vad_processor.process_frame(frame.clone()).await {
                    warn!("VAD processing error: {err}");
                }
            }
            Some(event) = vad_rx.recv() => {
                handle_vad_event(
                    event,
                    sample_rate,
                    &mut collecting,
                    &mut collected_frames,
                    &metrics,
                    &mut stt_engine,
                    &nlu_engine,
                    &stt_tx,
                    &nlu_tx,
                ).await;
            }
            else => {
                info!("Audio ingest channel closed, terminating pipeline");
                break;
            }
        }
    }
}

async fn handle_vad_event(
    event: VadEvent,
    sample_rate: u32,
    collecting: &mut bool,
    collected_frames: &mut Vec<AudioFrame>,
    metrics: &Arc<Mutex<PipelineMetrics>>,
    stt_engine: &mut WhisperEngine,
    nlu_engine: &NluEngine,
    stt_tx: &mpsc::UnboundedSender<Transcript>,
    nlu_tx: &mpsc::UnboundedSender<NluResult>,
) {
    match event {
        VadEvent::SpeechStart {
            timestamp,
            confidence,
        } => {
            debug!(?timestamp, confidence, "Speech start detected");
            *collecting = true;
            collected_frames.clear();
        }
        VadEvent::SpeechFrame { timestamp, pcm, .. } => {
            if !*collecting {
                *collecting = true;
            }

            let pcm_len = pcm.len();
            let frame = AudioFrame {
                data: Arc::from(pcm.into_boxed_slice()),
                timestamp,
                duration: Duration::from_secs_f32(pcm_len as f32 / sample_rate as f32),
                sample_rate,
            };
            collected_frames.push(frame);
        }
        VadEvent::SpeechEnd {
            timestamp,
            duration,
        } => {
            debug!(?timestamp, ?duration, "Speech end detected");
            if *collecting && !collected_frames.is_empty() {
                let frames = std::mem::take(collected_frames);
                let chunk = AudioChunk::new(frames);

                {
                    let mut guard = metrics.lock().await;
                    guard.generated_chunks += 1;
                }

                match stt_engine.transcribe(chunk).await {
                    Ok(transcript) => {
                        {
                            let mut guard = metrics.lock().await;
                            guard.transcripts += 1;
                        }
                        let _ = stt_tx.send(transcript.clone());
                        dispatch_nlu(transcript, metrics, nlu_engine, nlu_tx).await;
                    }
                    Err(err) => {
                        error!("STT transcription failed: {err}");
                    }
                }
            }
            *collecting = false;
            collected_frames.clear();
        }
        VadEvent::Silence { .. } => {
            collected_frames.clear();
        }
    }
}

async fn dispatch_nlu(
    transcript: Transcript,
    metrics: &Arc<Mutex<PipelineMetrics>>,
    nlu_engine: &NluEngine,
    nlu_tx: &mpsc::UnboundedSender<NluResult>,
) {
    match nlu_engine.process(&transcript).await {
        Ok(result) => {
            if matches!(result.command_type, CommandType::Local(_)) {
                let mut guard = metrics.lock().await;
                guard.commands += 1;
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
        guard.transcripts += 1;
        guard.last_activity = Some(Instant::now());
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
}
