// crates/whisper_stt/src/streaming.rs
use futures::StreamExt;

pub struct StreamingWhisper {
    config: WhisperConfig,
    buffer: Arc<RwLock<Vec<f32>>>,
    min_chunk_size: usize,
}

impl StreamingWhisper {
    pub async fn new(config: WhisperConfig) -> KlarnetResult<Self> {
        Ok(Self {
            config,
            buffer: Arc::new(RwLock::new(Vec::new())),
            min_chunk_size: 16000, // 1 second minimum
        })
    }

    pub async fn transcribe_stream(&self, pcm: &[f32]) -> KlarnetResult<Transcript> {
        let mut buffer = self.buffer.write();
        buffer.extend_from_slice(pcm);

        if buffer.len() < self.min_chunk_size {
            // Not enough data yet, return empty transcript
            return Ok(Transcript {
                id: Uuid::new_v4(),
                language: self.config.language.clone(),
                segments: vec![],
                full_text: String::new(),
                processing_time: Duration::from_millis(0),
            });
        }

        // Process buffered data
        let data_to_process = buffer.clone();
        buffer.clear();

        // In production, stream to faster-whisper
        self.process_streaming(&data_to_process).await
    }

    async fn process_streaming(&self, pcm: &[f32]) -> KlarnetResult<Transcript> {
        // Mock streaming implementation
        // In production, use actual faster-whisper streaming API

        let segments = vec![TranscriptSegment {
            start: 0.0,
            end: pcm.len() as f64 / 16000.0,
            text: "streaming transcription".to_string(),
            confidence: 0.90,
            words: vec![],
        }];

        Ok(Transcript {
            id: Uuid::new_v4(),
            language: self.config.language.clone(),
            segments: segments.clone(),
            full_text: segments.iter().map(|s| s.text.clone()).collect::<Vec<_>>().join(" "),
            processing_time: Duration::from_millis(30),
        })
    }
}