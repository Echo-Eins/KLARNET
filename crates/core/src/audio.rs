// crates/core/src/audio.rs

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

/// Audio configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub buffer_size: usize,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            bits_per_sample: 16,
            buffer_size: 1024,
        }
    }
}

/// Audio frame
#[derive(Debug, Clone)]
pub struct AudioFrame {
    pub data: Arc<[f32]>,
    pub timestamp: DateTime<Utc>,
    pub duration: Duration,
    pub sample_rate: u32,
}

/// Audio chunk for processing
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub id: uuid::Uuid,
    pub frames: Vec<AudioFrame>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub total_samples: usize,
}

impl AudioChunk {
    pub fn new(frames: Vec<AudioFrame>) -> Self {
        let start_time = frames.first().map(|f| f.timestamp).unwrap_or_else(Utc::now);
        let end_time = frames.last().map(|f| f.timestamp).unwrap_or_else(Utc::now);
        let total_samples = frames.iter().map(|f| f.data.len()).sum();

        Self {
            id: uuid::Uuid::new_v4(),
            frames,
            start_time,
            end_time,
            total_samples,
        }
    }

    pub fn to_pcm(&self) -> Vec<f32> {
        self.frames
            .iter()
            .flat_map(|f| f.data.iter().copied())
            .collect()
    }
}