// crates/buffering/src/segment_manager.rs

use std::collections::HashMap;
use uuid::Uuid;

pub struct SegmentManager {
    config: BufferConfig,
    segments: Arc<RwLock<HashMap<Uuid, AudioSegment>>>,
    active_segment: Arc<RwLock<Option<Uuid>>>,
}

#[derive(Debug, Clone)]
pub struct AudioSegment {
    pub id: Uuid,
    pub frames: Vec<AudioFrame>,
    pub start_time: Instant,
    pub end_time: Option<Instant>,
    pub sample_count: usize,
    pub is_complete: bool,
}

impl SegmentManager {
    pub fn new(config: BufferConfig) -> Self {
        Self {
            config,
            segments: Arc::new(RwLock::new(HashMap::new())),
            active_segment: Arc::new(RwLock::new(None)),
        }
    }

    pub fn create_segment(&self) -> Uuid {
        let id = Uuid::new_v4();
        let segment = AudioSegment {
            id,
            frames: Vec::new(),
            start_time: Instant::now(),
            end_time: None,
            sample_count: 0,
            is_complete: false,
        };

        {
            let mut segments = self.segments.write();
            segments.insert(id, segment);
        }

        {
            let mut active = self.active_segment.write();
            *active = Some(id);
        }

        info!("Created new audio segment: {}", id);
        id
    }

    pub fn add_frame_to_segment(&self, frame: AudioFrame) -> KlarnetResult<()> {
        let segment_id = {
            let active = self.active_segment.read();
            active.ok_or_else(|| KlarnetError::Audio("No active segment".to_string()))?
        };

        let mut segments = self.segments.write();
        let segment = segments.get_mut(&segment_id)
            .ok_or_else(|| KlarnetError::Audio("Segment not found".to_string()))?;

        segment.frames.push(frame.clone());
        segment.sample_count += frame.data.len();

        // Check if segment exceeds max duration
        let duration_ms = segment.sample_count as f32 / self.config.sample_rate as f32 * 1000.0;
        if duration_ms > self.config.max_chunk_duration_ms as f32 {
            segment.is_complete = true;
            segment.end_time = Some(Instant::now());
            warn!("Segment {} exceeded max duration, marking complete", segment_id);
        }

        Ok(())
    }

    pub fn complete_segment(&self, id: Uuid) -> KlarnetResult<AudioSegment> {
        let mut segments = self.segments.write();
        let mut segment = segments.remove(&id)
            .ok_or_else(|| KlarnetError::Audio("Segment not found".to_string()))?;

        segment.is_complete = true;
        segment.end_time = Some(Instant::now());

        // Clear active segment if it matches
        {
            let mut active = self.active_segment.write();
            if *active == Some(id) {
                *active = None;
            }
        }

        let duration = segment.end_time.unwrap() - segment.start_time;
        info!("Completed segment {}: {:.2}s", id, duration.as_secs_f32());

        Ok(segment)
    }

    pub fn cleanup_old_segments(&self, max_age: Duration) {
        let mut segments = self.segments.write();
        let now = Instant::now();

        segments.retain(|id, segment| {
            let age = now - segment.start_time;
            if age > max_age && segment.is_complete {
                debug!("Removing old segment: {}", id);
                false
            } else {
                true
            }
        });
    }
}