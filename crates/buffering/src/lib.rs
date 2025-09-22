// crates/buffering/src/lib.rs

use klarnet_core::{AudioChunk, AudioFrame, KlarnetError, KlarnetResult};
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

pub mod ring_buffer;
pub mod segment_manager;
pub mod adaptive;

use ring_buffer::RingBuffer;
use segment_manager::SegmentManager;

/// Audio buffering configuration
#[derive(Debug, Clone)]
pub struct BufferConfig {
    pub sample_rate: u32,
    pub chunk_duration_ms: usize,
    pub max_chunk_duration_ms: usize,
    pub pre_roll_duration_ms: usize,
    pub post_roll_duration_ms: usize,
    pub max_buffer_size_mb: usize,
    pub adaptive_buffering: bool,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            chunk_duration_ms: 2000,
            max_chunk_duration_ms: 10000,
            pre_roll_duration_ms: 1000,
            post_roll_duration_ms: 500,
            max_buffer_size_mb: 100,
            adaptive_buffering: true,
        }
    }
}

/// Audio buffer for managing speech segments
pub struct AudioBuffer {
    config: BufferConfig,
    ring_buffer: Arc<RwLock<RingBuffer<f32>>>,
    segment_manager: SegmentManager,
    pre_roll_buffer: Arc<RwLock<VecDeque<AudioFrame>>>,
    current_segment: Arc<RwLock<Vec<AudioFrame>>>,
    metrics: Arc<RwLock<BufferMetrics>>,
}

#[derive(Debug, Default)]
struct BufferMetrics {
    total_frames_buffered: u64,
    total_segments_created: u64,
    current_buffer_size_bytes: usize,
    peak_buffer_size_bytes: usize,
    dropped_frames: u64,
}

impl AudioBuffer {
    pub fn new(sample_rate: u32) -> Self {
        let config = BufferConfig {
            sample_rate,
            ..Default::default()
        };

        Self::with_config(config)
    }

    pub fn with_config(config: BufferConfig) -> Self {
        let max_samples = (config.max_buffer_size_mb * 1024 * 1024) / std::mem::size_of::<f32>();
        let ring_buffer = Arc::new(RwLock::new(RingBuffer::new(max_samples)));
        let segment_manager = SegmentManager::new(config.clone());
        let pre_roll_capacity = (config.pre_roll_duration_ms as f32 / 1000.0 * config.sample_rate as f32) as usize;

        Self {
            config,
            ring_buffer,
            segment_manager,
            pre_roll_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(pre_roll_capacity))),
            current_segment: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(BufferMetrics::default())),
        }
    }

    /// Add audio frame to buffer
    pub fn add_frame(&self, frame: AudioFrame) {
        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.total_frames_buffered += 1;
            metrics.current_buffer_size_bytes += frame.data.len() * std::mem::size_of::<f32>();
            metrics.peak_buffer_size_bytes = metrics.peak_buffer_size_bytes.max(metrics.current_buffer_size_bytes);
        }

        // Add to ring buffer
        {
            let mut buffer = self.ring_buffer.write();
            for sample in frame.data.iter() {
                buffer.push(*sample);
            }
        }

        // Update pre-roll buffer
        {
            let mut pre_roll = self.pre_roll_buffer.write();
            pre_roll.push_back(frame.clone());

            // Keep pre-roll buffer size limited
            let max_pre_roll_frames = (self.config.pre_roll_duration_ms as f32 / 1000.0
                * self.config.sample_rate as f32 / 1024.0) as usize;

            while pre_roll.len() > max_pre_roll_frames {
                pre_roll.pop_front();
            }
        }

        // Add to current segment
        {
            let mut segment = self.current_segment.write();
            segment.push(frame);
        }
    }

    /// Start new segment
    pub fn start_segment(&self) -> Vec<AudioFrame> {
        debug!("Starting new audio segment");

        // Include pre-roll frames
        let pre_roll_frames = {
            let pre_roll = self.pre_roll_buffer.read();
            pre_roll.iter().cloned().collect::<Vec<_>>()
        };

        // Clear current segment
        {
            let mut segment = self.current_segment.write();
            segment.clear();
            segment.extend(pre_roll_frames.clone());
        }

        pre_roll_frames
    }

    /// End current segment and return audio chunk
    pub fn end_segment(&self) -> Option<AudioChunk> {
        debug!("Ending audio segment");

        let frames = {
            let mut segment = self.current_segment.write();
            if segment.is_empty() {
                return None;
            }
            std::mem::take(&mut *segment)
        };

        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.total_segments_created += 1;
        }

        Some(AudioChunk::new(frames))
    }

    /// Get buffered audio as continuous PCM
    pub fn get_buffered_pcm(&self, duration_ms: usize) -> Vec<f32> {
        let samples_needed = (duration_ms as f32 / 1000.0 * self.config.sample_rate as f32) as usize;

        let buffer = self.ring_buffer.read();
        buffer.get_last_n(samples_needed)
    }

    /// Clear all buffers
    pub fn clear(&self) {
        self.ring_buffer.write().clear();
        self.pre_roll_buffer.write().clear();
        self.current_segment.write().clear();

        let mut metrics = self.metrics.write();
        metrics.current_buffer_size_bytes = 0;
    }

    pub fn get_metrics(&self) -> BufferMetrics {
        self.metrics.read().clone()
    }
}