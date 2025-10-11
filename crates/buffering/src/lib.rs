use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use klarnet_core::{AudioChunk, AudioFrame};
use serde::{Deserialize, Serialize};
use tracing::debug;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferConfig {
    pub sample_rate: u32,
    pub pre_roll_ms: u64,
    pub max_chunk_ms: u64,
}

impl BufferConfig {
    pub fn new(sample_rate: u32, pre_roll_ms: u64, max_chunk_ms: u64) -> Self {
        Self {
            sample_rate,
            pre_roll_ms,
            max_chunk_ms,
        }
    }
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            pre_roll_ms: 500,
            max_chunk_ms: 12_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompletedSegment {
    pub chunk: AudioChunk,
    pub overflowed: bool,
}

pub struct SegmentCollector {
    config: BufferConfig,
    pre_roll_frames: VecDeque<AudioFrame>,
    pre_roll_total: Duration,
    active_frames: Vec<AudioFrame>,
    active_duration: Duration,
    max_chunk_duration: Duration,
    collecting: bool,
}

impl SegmentCollector {
    pub fn new(config: BufferConfig) -> Self {
        let max_chunk = Duration::from_millis(config.max_chunk_ms.max(config.pre_roll_ms));
        let pre_roll_capacity =
            ((config.sample_rate as u64 * config.pre_roll_ms) / 1000).max(1) as usize;

        Self {
            config,
            pre_roll_frames: VecDeque::with_capacity(pre_roll_capacity),
            pre_roll_total: Duration::from_millis(0),
            active_frames: Vec::new(),
            active_duration: Duration::from_millis(0),
            max_chunk_duration: max_chunk,
            collecting: false,
        }
    }

    pub fn observe_frame(&mut self, frame: &AudioFrame) {
        if self.collecting {
            return;
        }

        self.pre_roll_frames.push_back(frame.clone());
        self.pre_roll_total += frame.duration;

        while self.pre_roll_total > Duration::from_millis(self.config.pre_roll_ms) {
            if let Some(oldest) = self.pre_roll_frames.pop_front() {
                self.pre_roll_total = self.pre_roll_total.saturating_sub(oldest.duration);
            } else {
                self.pre_roll_total = Duration::from_millis(0);
                break;
            }
        }
    }

    pub fn start(&mut self) {
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

    pub fn push_frame(&mut self, frame: AudioFrame) -> Vec<CompletedSegment> {
        if !self.collecting {
            self.start();
        }

        self.active_duration += frame.duration;
        self.active_frames.push(frame);

        if self.active_duration >= self.max_chunk_duration {
            vec![self.finish_internal(true)]
        } else {
            Vec::new()
        }
    }

    pub fn push_pcm(&mut self, timestamp: DateTime<Utc>, pcm: Vec<f32>) -> Vec<CompletedSegment> {
        let duration = if pcm.is_empty() {
            Duration::from_millis(0)
        } else {
            Duration::from_secs_f32(pcm.len() as f32 / self.config.sample_rate as f32)
        };

        let frame = AudioFrame {
            data: Arc::from(pcm.into_boxed_slice()),
            timestamp,
            duration,
            sample_rate: self.config.sample_rate,
            channels: self
                .active_frames
                .first()
                .or_else(|| self.pre_roll_frames.front())
                .map(|frame| frame.channels)
                .unwrap_or(1),
        };

        self.push_frame(frame)
    }

    pub fn on_silence(&mut self) {
        if self.collecting {
            return;
        }
        self.pre_roll_frames.clear();
        self.pre_roll_total = Duration::from_millis(0);
    }

    pub fn finish(&mut self) -> Option<CompletedSegment> {
        if !self.collecting {
            return None;
        }

        self.collecting = false;
        if self.active_frames.is_empty() {
            self.reset();
            return None;
        }

        let completed = self.finish_internal(false);
        self.reset();
        Some(completed)
    }

    pub fn reset(&mut self) {
        self.collecting = false;
        self.active_frames.clear();
        self.active_duration = Duration::from_millis(0);
        self.pre_roll_frames.clear();
        self.pre_roll_total = Duration::from_millis(0);
    }

    fn finish_internal(&mut self, overflowed: bool) -> CompletedSegment {
        let frames = std::mem::take(&mut self.active_frames);
        self.active_duration = Duration::from_millis(0);
        debug!(
            frame_count = frames.len(),
            overflowed, "Completing buffered segment"
        );
        CompletedSegment {
            chunk: AudioChunk::new(frames),
            overflowed,
        }
    }
}
