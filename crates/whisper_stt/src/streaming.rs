// crates/whisper_stt/src/streaming.rs

use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use chrono::Utc;

use parking_lot::RwLock;
use uuid::Uuid;
use tokio::sync::Mutex;
use klarnet_core::{AudioChunk, AudioFrame, KlarnetResult, Transcript};

use crate::{WhisperConfig, WhisperEngine};

pub struct StreamingWhisper {
    config: WhisperConfig,
    buffer: Arc<RwLock<Vec<f32>>>,
    processed_samples: Arc<RwLock<usize>>,
    min_chunk_size: usize,
    engine: Arc<Mutex<WhisperEngine>>,
}

impl StreamingWhisper {
    pub async fn new(config: WhisperConfig) -> KlarnetResult<Self> {
        const SAMPLE_RATE: usize = 16_000;
        const MIN_CHUNK_DURATION_MS: u64 = 500;

        let min_chunk_size = ((SAMPLE_RATE as u64 * MIN_CHUNK_DURATION_MS) / 1_000) as usize;

        let engine = WhisperEngine::new(config.clone()).await?;

        Ok(Self {
            config,
            buffer: Arc::new(RwLock::new(Vec::new())),
            processed_samples: Arc::new(RwLock::new(0)),
            min_chunk_size,
            engine: Arc::new(Mutex::new(engine)),
        })
    }

    pub async fn transcribe_stream(&self, pcm: &[f32]) -> KlarnetResult<Transcript> {
        const SAMPLE_RATE: f64 = 16_000.0;

        let data_to_process = {
            let mut buffer = self.buffer.write();
            buffer.extend_from_slice(pcm);

            if buffer.len() < self.min_chunk_size {
                return Ok(Transcript {
                    id: Uuid::new_v4(),
                    language: self.config.language.clone(),
                    segments: vec![],
                    full_text: String::new(),
                    processing_time: Duration::from_millis(0),
                });
            }

            buffer.clone()
        };

        let chunk = Self::build_chunk(&data_to_process);

        let mut engine = self.engine.lock().await;
        let mut transcript = engine.transcribe(chunk).await?;
        drop(engine);

        let processed_samples = transcript
            .segments
            .last()
            .map(|segment| (segment.end * SAMPLE_RATE).ceil() as usize)
            .unwrap_or(0);

        let offset_samples = { *self.processed_samples.read() };
        let offset_seconds = offset_samples as f64 / SAMPLE_RATE;

        if offset_seconds > 0.0 {
            for segment in &mut transcript.segments {
                segment.start += offset_seconds;
                segment.end += offset_seconds;

                for word in &mut segment.words {
                    word.start += offset_seconds;
                    word.end += offset_seconds;
                }
            }
        }

        if processed_samples > 0 {
            {
                let mut processed = self.processed_samples.write();
                *processed += processed_samples;
            }

            let mut buffer = self.buffer.write();
            if processed_samples >= buffer.len() {
                buffer.clear();
            } else {
                buffer.drain(..processed_samples);
            }
        }

        Ok(transcript)
    }

    fn build_chunk(pcm: &[f32]) -> AudioChunk {
        const SAMPLE_RATE: u32 = 16_000;

        let timestamp = Utc::now();
        let duration = Duration::from_secs_f64(pcm.len() as f64 / f64::from(SAMPLE_RATE));
        let frame = AudioFrame {
            data: Arc::from(pcm.to_vec().into_boxed_slice()),
            timestamp,
            duration,
            sample_rate: SAMPLE_RATE,
        };

        AudioChunk::new(vec![frame])
    }
}
