// crates/vad/src/lib.rs

use async_trait::async_trait;
use klarnet_core::{AudioFrame, KlarnetError, KlarnetResult, VadEvent};
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

pub mod webrtc;
pub mod energy;
pub mod hybrid;

pub use webrtc::WebRtcVad;
pub use energy::EnergyVad;
pub use hybrid::HybridVad;

/// VAD configuration
#[derive(Debug, Clone)]
pub struct VadConfig {
    pub mode: VadMode,
    pub aggressiveness: u8, // 0-3 for WebRTC
    pub frame_duration_ms: usize, // 10, 20, or 30 ms for WebRTC
    pub min_speech_duration_ms: usize,
    pub min_silence_duration_ms: usize,
    pub energy_threshold: f32,
    pub speech_pad_ms: usize, // Padding before/after speech
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            mode: VadMode::WebRtc,
            aggressiveness: 2,
            frame_duration_ms: 30,
            min_speech_duration_ms: 200,
            min_silence_duration_ms: 500,
            energy_threshold: 0.01,
            speech_pad_ms: 300,
        }
    }
}

#[derive(Debug, Clone)]
pub enum VadMode {
    WebRtc,
    Energy,
    Hybrid,
}

/// VAD trait
#[async_trait]
pub trait VoiceActivityDetector: Send + Sync {
    async fn process(&mut self, frame: AudioFrame) -> KlarnetResult<Option<VadEvent>>;
    fn reset(&mut self);
}

/// VAD processor managing the detection pipeline
pub struct VadProcessor {
    config: VadConfig,
    detector: Box<dyn VoiceActivityDetector>,
    state: Arc<RwLock<VadState>>,
    tx: mpsc::UnboundedSender<VadEvent>,
}

#[derive(Debug, Clone)]
struct VadState {
    is_speaking: bool,
    speech_start: Option<Instant>,
    silence_start: Option<Instant>,
    last_speech_end: Option<Instant>,
    buffered_frames: Vec<AudioFrame>,
}

impl VadProcessor {
    pub fn new(config: VadConfig) -> KlarnetResult<(Self, mpsc::UnboundedReceiver<VadEvent>)> {
        let detector: Box<dyn VoiceActivityDetector> = match config.mode {
            VadMode::WebRtc => Box::new(WebRtcVad::new(config.clone())?),
            VadMode::Energy => Box::new(EnergyVad::new(config.clone())),
            VadMode::Hybrid => Box::new(HybridVad::new(config.clone())?),
        };

        let (tx, rx) = mpsc::unbounded_channel();

        let state = Arc::new(RwLock::new(VadState {
            is_speaking: false,
            speech_start: None,
            silence_start: None,
            last_speech_end: None,
            buffered_frames: Vec::new(),
        }));

        Ok((
            Self {
                config,
                detector,
                state,
                tx,
            },
            rx,
        ))
    }

    pub async fn process_frame(&mut self, frame: AudioFrame) -> KlarnetResult<()> {
        let event = self.detector.process(frame.clone()).await?;

        if let Some(event) = event {
            self.handle_event(event, frame).await?;
        }

        Ok(())
    }

    async fn handle_event(&mut self, event: VadEvent, frame: AudioFrame) -> KlarnetResult<()> {
        let mut state = self.state.write();

        match event {
            VadEvent::SpeechStart { timestamp, confidence } => {
                if !state.is_speaking {
                    state.is_speaking = true;
                    state.speech_start = Some(Instant::now());
                    state.silence_start = None;

                    // Send buffered frames (pre-roll)
                    for buffered_frame in &state.buffered_frames {
                        let _ = self.tx.send(VadEvent::SpeechFrame {
                            timestamp: buffered_frame.timestamp,
                            pcm: buffered_frame.data.clone(),
                            energy: self.calculate_energy(&buffered_frame.data),
                        });
                    }
                    state.buffered_frames.clear();

                    info!("Speech started (confidence: {:.2})", confidence);
                    let _ = self.tx.send(VadEvent::SpeechStart { timestamp, confidence });
                }
            }

            VadEvent::SpeechFrame { .. } => {
                if state.is_speaking {
                    state.silence_start = None;
                    let _ = self.tx.send(VadEvent::SpeechFrame {
                        timestamp: frame.timestamp,
                        pcm: frame.data.clone(),
                        energy: self.calculate_energy(&frame.data),
                    });
                }
            }

            VadEvent::Silence { timestamp, .. } => {
                if state.is_speaking {
                    if state.silence_start.is_none() {
                        state.silence_start = Some(Instant::now());
                    }

                    if let Some(silence_start) = state.silence_start {
                        let silence_duration = Instant::now() - silence_start;

                        if silence_duration > Duration::from_millis(self.config.min_silence_duration_ms as u64) {
                            state.is_speaking = false;
                            state.last_speech_end = Some(Instant::now());

                            if let Some(speech_start) = state.speech_start {
                                let duration = Instant::now() - speech_start;
                                info!("Speech ended (duration: {:.2}s)", duration.as_secs_f32());

                                let _ = self.tx.send(VadEvent::SpeechEnd {
                                    timestamp,
                                    duration,
                                });
                            }

                            state.speech_start = None;
                            state.silence_start = None;
                        }
                    }
                } else {
                    // Buffer frames for pre-roll
                    state.buffered_frames.push(frame.clone());

                    // Keep buffer size limited
                    let max_buffer_size = (self.config.speech_pad_ms as f32 / 1000.0 * frame.sample_rate as f32) as usize;
                    while state.buffered_frames.len() > max_buffer_size / 1024 {
                        state.buffered_frames.remove(0);
                    }
                }
            }

            _ => {}
        }

        Ok(())
    }

    fn calculate_energy(&self, pcm: &[f32]) -> f32 {
        let sum: f32 = pcm.iter().map(|s| s * s).sum();
        (sum / pcm.len() as f32).sqrt()
    }
}
