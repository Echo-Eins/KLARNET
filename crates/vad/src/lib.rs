use std::collections::VecDeque;
use std::time::Duration as StdDuration;

use chrono::{DateTime, Duration as ChronoDuration, Utc};

use klarnet_core::{AudioFrame, KlarnetResult, VadEvent};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{debug, trace};

const EPSILON: f32 = 1e-6;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VadMode {
    Energy,
}

impl Default for VadMode {
    fn default() -> Self {
        Self::Energy
    }
}

fn default_frame_duration_ms() -> usize {
    30
}

fn default_energy_threshold() -> f32 {
    0.01
}

fn default_initial_noise_floor() -> f32 {
    0.001
}

fn default_noise_update_rate() -> f32 {
    0.05
}

fn default_min_speech_duration_ms() -> usize {
    150
}

fn default_min_silence_duration_ms() -> usize {
    300
}

fn default_hysteresis_ratio() -> f32 {
    0.2
}

fn default_speech_pad_ms() -> usize {
    120
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadConfig {
    #[serde(default)]
    pub mode: VadMode,
    #[serde(default = "default_frame_duration_ms")]
    pub frame_duration_ms: usize,
    #[serde(default = "default_energy_threshold")]
    pub energy_threshold: f32,
    #[serde(default = "default_initial_noise_floor")]
    pub initial_noise_floor: f32,
    #[serde(default = "default_noise_update_rate")]
    pub noise_update_rate: f32,
    #[serde(default = "default_min_speech_duration_ms")]
    pub min_speech_duration_ms: usize,
    #[serde(default = "default_min_silence_duration_ms")]
    pub min_silence_duration_ms: usize,
    #[serde(default = "default_hysteresis_ratio")]
    pub hysteresis_ratio: f32,
    #[serde(default = "default_speech_pad_ms")]
    pub speech_pad_ms: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            mode: VadMode::default(),
            frame_duration_ms: default_frame_duration_ms(),
            energy_threshold: default_energy_threshold(),
            initial_noise_floor: default_initial_noise_floor(),
            noise_update_rate: default_noise_update_rate(),
            min_speech_duration_ms: default_min_speech_duration_ms(),
            min_silence_duration_ms: default_min_silence_duration_ms(),
            hysteresis_ratio: default_hysteresis_ratio(),
            speech_pad_ms: default_speech_pad_ms(),
        }
    }
}

impl VadConfig {
    fn min_speech_frames(&self) -> usize {
        (self.min_speech_duration_ms + self.frame_duration_ms - 1) / self.frame_duration_ms
    }

    fn min_silence_frames(&self) -> usize {
        (self.min_silence_duration_ms + self.frame_duration_ms - 1) / self.frame_duration_ms
    }

    fn pad_frames(&self) -> usize {
        (self.speech_pad_ms + self.frame_duration_ms - 1) / self.frame_duration_ms
    }
}

#[derive(Debug, Clone)]
struct BufferFrame {
    timestamp: DateTime<Utc>,
    pcm: Vec<f32>,
    energy: f32,
}

#[derive(Debug, Default)]
struct VadMetrics {
    frames_processed: usize,
    speech_frames: usize,
    silence_frames: usize,
    speech_segments: usize,
}

#[derive(Debug)]
struct VadState {
    in_speech: bool,
    speech_start: Option<DateTime<Utc>>,
    last_speech_timestamp: Option<DateTime<Utc>>,
    last_speech_duration: Option<StdDuration>,
    speech_frame_counter: usize,
    silence_frame_counter: usize,
    noise_floor: f32,
    current_energy: f32,
    pending_silence: StdDuration,
    buffer: VecDeque<BufferFrame>,
}

impl VadState {
    fn new(config: &VadConfig) -> Self {
        Self {
            in_speech: false,
            speech_start: None,
            last_speech_timestamp: None,
            last_speech_duration: None,
            speech_frame_counter: 0,
            silence_frame_counter: 0,
            noise_floor: config.initial_noise_floor.max(EPSILON),
            current_energy: 0.0,
            pending_silence: StdDuration::default(),
            buffer: VecDeque::with_capacity(config.pad_frames().max(1)),
        }
    }

    fn push_buffered_frame(&mut self, frame: BufferFrame, limit: usize) {
        if limit == 0 {
            return;
        }
        self.buffer.push_back(frame);
        while self.buffer.len() > limit {
            self.buffer.pop_front();
        }
    }

    fn clear_buffer(&mut self) -> Vec<BufferFrame> {
        self.buffer.drain(..).collect()
    }
}

pub struct VadProcessor {
    config: VadConfig,
    tx: mpsc::UnboundedSender<VadEvent>,
    state: VadState,
    metrics: VadMetrics,
}

impl VadProcessor {
    pub fn new(config: VadConfig) -> KlarnetResult<(Self, mpsc::UnboundedReceiver<VadEvent>)> {
        let (tx, rx) = mpsc::unbounded_channel();
        let state = VadState::new(&config);
        let processor = Self {
            config,
            tx,
            state,
            metrics: VadMetrics::default(),
        };
        Ok((processor, rx))
    }

    pub async fn process_frame(&mut self, frame: AudioFrame) -> KlarnetResult<()> {
        self.metrics.frames_processed += 1;

        let pcm: Vec<f32> = frame.data.iter().copied().collect();
        let energy = if pcm.is_empty() {
            0.0
        } else {
            pcm.iter().map(|sample| sample * sample).sum::<f32>() / pcm.len() as f32
        };

        self.state.current_energy = energy;
        let timestamp = frame.timestamp;
        let duration = frame.duration;

        if !self.state.in_speech {
            self.state.noise_floor = (1.0 - self.config.noise_update_rate) * self.state.noise_floor
                + self.config.noise_update_rate * energy;
            self.state.noise_floor = self.state.noise_floor.max(EPSILON);
        }

        let speech_threshold = (self.state.noise_floor + self.config.energy_threshold)
            .max(self.config.energy_threshold);
        let release_threshold =
            (speech_threshold * (1.0 - self.config.hysteresis_ratio)).max(self.state.noise_floor);

        let mut classify_as_speech = false;
        if energy >= speech_threshold {
            self.state.speech_frame_counter += 1;
            self.state.silence_frame_counter = 0;
            classify_as_speech = true;
        } else if energy <= release_threshold {
            self.state.silence_frame_counter += 1;
            self.state.speech_frame_counter = 0;
            self.state.pending_silence += duration;
        } else if self.state.in_speech {
            classify_as_speech = true;
        }

        trace!(
            ?timestamp,
            energy,
            noise_floor = self.state.noise_floor,
            speech_threshold,
            release_threshold,
            classify_as_speech,
            in_speech = self.state.in_speech,
            "processed frame"
        );

        if !self.state.in_speech
            && self.state.speech_frame_counter >= self.config.min_speech_frames()
        {
            self.start_speech(timestamp, energy, speech_threshold, release_threshold);
            classify_as_speech = true;
        }

        if self.state.in_speech
            && self.state.silence_frame_counter >= self.config.min_silence_frames()
        {
            self.end_speech(timestamp);
            if self.state.pending_silence > StdDuration::default() {
                self.emit_silence(timestamp, self.state.pending_silence);
                self.state.pending_silence = StdDuration::default();
            }
            self.state.silence_frame_counter = 0;
        }

        if self.state.in_speech && classify_as_speech {
            self.emit_speech_frame(timestamp, pcm.clone(), energy);
            self.state.last_speech_timestamp = Some(timestamp);
            self.state.last_speech_duration = Some(duration);
            self.metrics.speech_frames += 1;
            self.state.pending_silence = StdDuration::default();
        } else if !self.state.in_speech {
            self.state.push_buffered_frame(
                BufferFrame {
                    timestamp,
                    pcm,
                    energy,
                },
                self.config.pad_frames(),
            );
            if self.state.silence_frame_counter >= self.config.min_silence_frames() {
                self.emit_silence(timestamp, self.state.pending_silence);
                self.state.pending_silence = StdDuration::default();
                self.state.silence_frame_counter = 0;
            }
            self.metrics.silence_frames += 1;
        }

        Ok(())
    }

    fn start_speech(
        &mut self,
        timestamp: DateTime<Utc>,
        energy: f32,
        speech_threshold: f32,
        release_threshold: f32,
    ) {
        self.state.in_speech = true;
        self.state.speech_start = Some(timestamp);
        self.state.silence_frame_counter = 0;
        self.metrics.speech_segments += 1;

        let confidence = ((energy - release_threshold)
            / (speech_threshold - release_threshold + EPSILON))
            .clamp(0.0, 1.0);

        debug!(
            ?timestamp,
            confidence, energy, speech_threshold, release_threshold, "speech start"
        );

        let _ = self.tx.send(VadEvent::SpeechStart {
            timestamp,
            confidence,
        });

        for buffered in self.state.clear_buffer() {
            self.emit_speech_frame(buffered.timestamp, buffered.pcm, buffered.energy);
        }
    }

    fn end_speech(&mut self, timestamp: DateTime<Utc>) {
        let speech_start = self.state.speech_start.take();
        self.state.in_speech = false;
        self.state.speech_frame_counter = 0;

        if let Some(start) = speech_start {
            let last_ts = self.state.last_speech_timestamp.unwrap_or(timestamp);
            let last_duration = self
                .state
                .last_speech_duration
                .unwrap_or_else(StdDuration::default);
            let offset = ChronoDuration::from_std(last_duration)
                .unwrap_or_else(|_| ChronoDuration::milliseconds(0));
            let total = (last_ts - start) + offset;
            if let Ok(duration) = total.to_std() {
                debug!(?timestamp, ?duration, "speech end");
                let _ = self.tx.send(VadEvent::SpeechEnd {
                    timestamp,
                    duration,
                });
            }
        }

        self.state.buffer.clear();
        self.state.last_speech_timestamp = None;
        self.state.last_speech_duration = None;
    }

    fn emit_speech_frame(&self, timestamp: DateTime<Utc>, pcm: Vec<f32>, energy: f32) {
        let _ = self.tx.send(VadEvent::SpeechFrame {
            timestamp,
            pcm,
            energy,
        });
    }

    fn emit_silence(&self, timestamp: DateTime<Utc>, duration: StdDuration) {
        if duration.is_zero() {
            return;
        }
        debug!(?timestamp, ?duration, "silence");
        let _ = self.tx.send(VadEvent::Silence {
            timestamp,
            duration,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::mpsc;

    fn build_frame(
        timestamp: DateTime<Utc>,
        duration: StdDuration,
        amplitude: f32,
        samples: usize,
    ) -> AudioFrame {
        let pcm: Vec<f32> = vec![amplitude; samples];
        AudioFrame {
            data: Arc::from(pcm.into_boxed_slice()),
            timestamp,
            duration,
            sample_rate: 16_000,
        }
    }

    async fn collect_events(
        processor: &mut VadProcessor,
        mut rx: mpsc::UnboundedReceiver<VadEvent>,
        frames: Vec<AudioFrame>,
    ) -> Vec<VadEvent> {
        for frame in frames {
            processor.process_frame(frame).await.unwrap();
        }

        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }
        events
    }

    #[tokio::test]
    async fn detects_silence() {
        let config = VadConfig {
            energy_threshold: 0.02,
            ..Default::default()
        };
        let (mut processor, rx) = VadProcessor::new(config).unwrap();

        let duration = StdDuration::from_millis(30);
        let timestamp = Utc::now();
        let frames = (0..5)
            .map(|i| {
                build_frame(
                    timestamp + ChronoDuration::milliseconds(i as i64 * 30),
                    duration,
                    0.001,
                    480,
                )
            })
            .collect::<Vec<_>>();

        let events = collect_events(&mut processor, rx, frames).await;
        assert!(events
            .iter()
            .all(|event| matches!(event, VadEvent::Silence { .. })));
    }

    #[tokio::test]
    async fn detects_speech_sequence() {
        let mut config = VadConfig::default();
        config.min_speech_duration_ms = 60;
        config.min_silence_duration_ms = 60;
        config.speech_pad_ms = 60;
        config.energy_threshold = 0.001;

        let (mut processor, rx) = VadProcessor::new(config).unwrap();

        let duration = StdDuration::from_millis(30);
        let timestamp = Utc::now();
        let mut frames = Vec::new();
        // Silence
        for i in 0..3 {
            frames.push(build_frame(
                timestamp + ChronoDuration::milliseconds(i as i64 * 30),
                duration,
                0.0005,
                480,
            ));
        }
        // Speech
        for i in 3..7 {
            frames.push(build_frame(
                timestamp + ChronoDuration::milliseconds(i as i64 * 30),
                duration,
                0.1,
                480,
            ));
        }
        // Silence tail
        for i in 7..11 {
            frames.push(build_frame(
                timestamp + ChronoDuration::milliseconds(i as i64 * 30),
                duration,
                0.0005,
                480,
            ));
        }

        let events = collect_events(&mut processor, rx, frames).await;
        let mut saw_start = false;
        let mut speech_frames = 0;
        let mut saw_end = false;
        let mut saw_silence = false;

        for event in events {
            match event {
                VadEvent::SpeechStart { .. } => saw_start = true,
                VadEvent::SpeechFrame { energy, pcm, .. } => {
                    assert!(energy > 0.0);
                    assert!(!pcm.is_empty());
                    speech_frames += 1;
                }
                VadEvent::SpeechEnd { duration, .. } => {
                    assert!(duration >= StdDuration::from_millis(60));
                    saw_end = true;
                }
                VadEvent::Silence { duration, .. } => {
                    if duration >= StdDuration::from_millis(60) {
                        saw_silence = true;
                    }
                }
            }
        }

        assert!(saw_start);
        assert!(speech_frames >= 3);
        assert!(saw_end);
        assert!(saw_silence);
    }

    #[tokio::test]
    async fn ignores_noise_below_threshold() {
        let mut config = VadConfig::default();
        config.energy_threshold = 0.05;
        config.min_speech_duration_ms = 90;

        let (mut processor, rx) = VadProcessor::new(config).unwrap();

        let duration = StdDuration::from_millis(30);
        let timestamp = Utc::now();
        let frames = (0..10)
            .map(|i| {
                build_frame(
                    timestamp + ChronoDuration::milliseconds(i as i64 * 30),
                    duration,
                    0.02,
                    480,
                )
            })
            .collect::<Vec<_>>();

        let events = collect_events(&mut processor, rx, frames).await;
        assert!(events
            .iter()
            .all(|event| matches!(event, VadEvent::Silence { .. })));
    }
}
