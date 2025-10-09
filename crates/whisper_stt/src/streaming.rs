use std::sync::Arc;
use chrono::Utc;
use klarnet_core::{AudioChunk, AudioFrame, KlarnetResult, Transcript};

use crate::{WhisperConfig, WhisperEngine};
const STREAM_SAMPLE_RATE: u32 = 16_000;
const MIN_CHUNK_SAMPLES: usize = (STREAM_SAMPLE_RATE as usize) / 2;

pub struct StreamingWhisper {
    engine: WhisperEngine,
    buffer: Vec<f32>,
    min_chunk: usize,
}
impl StreamingWhisper {
    pub async fn new(config: WhisperConfig) -> KlarnetResult<Self> {
        let engine = WhisperEngine::new(config).await?;
        Ok(Self {
            engine,
            buffer: Vec::new(),
            min_chunk: MIN_CHUNK_SAMPLES,
        })
    }
    pub async fn push_audio(&mut self, pcm: &[f32]) -> KlarnetResult<Option<Transcript>> {
        self.buffer.extend_from_slice(pcm);
        if self.buffer.len() < self.min_chunk {
            return Ok(None);
        }
        let chunk = Self::build_chunk(&self.buffer);
        let transcript = self.engine.transcribe(chunk).await?;
        self.buffer.clear();
        Ok(Some(transcript))
    }

    fn build_chunk(pcm: &[f32]) -> AudioChunk {
        let duration =
            std::time::Duration::from_secs_f32(pcm.len() as f32 / STREAM_SAMPLE_RATE as f32);
        let frame = AudioFrame {
            data: Arc::from(pcm.to_vec().into_boxed_slice()),
            timestamp: Utc::now(),
            duration,
            sample_rate: STREAM_SAMPLE_RATE,
        };

        AudioChunk::new(vec![frame])
    }
}