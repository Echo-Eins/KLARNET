use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::WebSocket;
use bytes::Bytes;
use chrono::Utc;
use klarnet_core::{AudioChunk, AudioFrame, KlarnetError, KlarnetResult, Transcript};
use klarnet_observability::metrics::MetricType;
use klarnet_observability::MetricsCollector;
use nlu::NluEngine;
use parking_lot::RwLock;
use serde_json::json;
use tokio::sync::Mutex;
use tracing::{error, info};
use uuid::Uuid;
use whisper_stt::WhisperEngine;

pub(crate) const STREAM_CHUNK_SAMPLES: usize = 32_000;

pub struct ApiHandlers {
    whisper: Option<Arc<Mutex<WhisperEngine>>>,
    nlu: Option<Arc<NluEngine>>,
    metrics: Arc<MetricsCollector>,
    sessions: Arc<RwLock<SessionManager>>,
}

impl ApiHandlers {
    pub fn new(metrics: Arc<MetricsCollector>) -> Self {
        Self {
            whisper: None,
            nlu: None,
            metrics,
            sessions: Arc::new(RwLock::new(SessionManager::new())),
        }
    }

    pub fn with_engines(
        whisper: Arc<Mutex<WhisperEngine>>,
        nlu: Arc<NluEngine>,
        metrics: Arc<MetricsCollector>,
    ) -> Self {
        Self {
            whisper: Some(whisper),
            nlu: Some(nlu),
            metrics,
            sessions: Arc::new(RwLock::new(SessionManager::new())),
        }
    }

    pub fn whisper_engine(&self) -> Option<Arc<Mutex<WhisperEngine>>> {
        self.whisper.as_ref().map(Arc::clone)
    }

    pub async fn transcribe_file(
        &self,
        audio_data: Bytes,
        sample_rate: u32,
    ) -> KlarnetResult<Transcript> {
        let whisper = self
            .whisper_engine()
            .ok_or_else(|| KlarnetError::Stt("STT engine not initialised".to_string()))?;

        let pcm = decode_pcm_s16le(&audio_data)?;
        let chunk = build_chunk_from_pcm(&pcm, sample_rate);

        let mut engine = whisper.lock().await;
        let transcript = engine.transcribe(chunk).await?;
        self.metrics.increment(MetricType::TranscriptionsCompleted);
        Ok(transcript)
    }

    pub async fn interpret_text(&self, text: String) -> KlarnetResult<serde_json::Value> {
        let nlu = self
            .nlu
            .as_ref()
            .ok_or_else(|| KlarnetError::Nlu("NLU engine not initialised".to_string()))?;

        let transcript = Transcript {
            id: Uuid::new_v4(),
            language: "ru".to_string(),
            segments: vec![],
            full_text: text,
            processing_time: Duration::from_millis(0),
        };

        let result = nlu.process(&transcript).await?;
        Ok(json!({
            "transcript": result.transcript,
            "intent": result.intent,
            "wake_word_detected": result.wake_word_detected,
            "command_type": result.command_type,
        }))
    }

    pub async fn chat(&self, request: super::ChatRequest) -> KlarnetResult<super::ChatResponse> {
        if let Some(nlu) = &self.nlu {
            let transcript = Transcript {
                id: Uuid::new_v4(),
                language: "ru".to_string(),
                segments: vec![],
                full_text: request.message.clone(),
                processing_time: Duration::from_millis(0),
            };

            let result = nlu.process(&transcript).await?;
            Ok(super::ChatResponse {
                response: format!("Обработано: {}", result.transcript),
                action: None,
            })
        } else {
            Ok(super::ChatResponse {
                response: "Chat processing not available".to_string(),
                action: None,
            })
        }
    }

    pub fn metrics_snapshot(&self) -> String {
        self.metrics.get_prometheus_metrics()
    }

    pub(crate) fn create_session(&self) -> Uuid {
        self.sessions.write().create_session()
    }

    pub(crate) fn append_session_audio(
        &self,
        session_id: &Uuid,
        samples: &[f32],
    ) -> Option<Vec<f32>> {
        self.sessions.write().append_audio(session_id, samples)
    }

    pub(crate) fn cleanup_sessions(&self, max_age: Duration) {
        self.sessions.write().cleanup(max_age);
    }
}

#[derive(Default)]
struct SessionManager {
    sessions: HashMap<Uuid, Session>,
}

impl SessionManager {
    fn new() -> Self {
        Self::default()
    }

    fn create_session(&mut self) -> Uuid {
        let id = Uuid::new_v4();
        self.sessions.insert(
            id,
            Session {
                last_activity: Utc::now(),
                audio_buffer: Vec::new(),
            },
        );
        id
    }

    fn append_audio(&mut self, session_id: &Uuid, samples: &[f32]) -> Option<Vec<f32>> {
        let session = self.sessions.get_mut(session_id)?;
        session.last_activity = Utc::now();
        session.audio_buffer.extend_from_slice(samples);

        if session.audio_buffer.len() >= STREAM_CHUNK_SAMPLES {
            let mut buffer = Vec::new();
            std::mem::swap(&mut buffer, &mut session.audio_buffer);
            Some(buffer)
        } else {
            None
        }
    }

    fn cleanup(&mut self, max_age: Duration) {
        let now = Utc::now();
        self.sessions.retain(|_, session| {
            now - session.last_activity < chrono::Duration::from_std(max_age).unwrap()
        });
    }
}

struct Session {
    last_activity: chrono::DateTime<Utc>,
    audio_buffer: Vec<f32>,
}

pub(crate) fn decode_pcm_s16le(data: &[u8]) -> KlarnetResult<Vec<f32>> {
    if data.len() % 2 != 0 {
        return Err(KlarnetError::Audio(
            "PCM payload length must be an even number of bytes".to_string(),
        ));
    }

    Ok(data
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
        .collect())
}

pub(crate) fn build_chunk_from_pcm(samples: &[f32], sample_rate: u32) -> AudioChunk {
    let duration = if samples.is_empty() {
        Duration::from_millis(0)
    } else {
        Duration::from_secs_f64(samples.len() as f64 / sample_rate as f64)
    };

    let frame = AudioFrame {
        data: Arc::from(samples.to_vec().into_boxed_slice()),
        timestamp: Utc::now(),
        duration,
        sample_rate,
        channels: 1,
    };

    AudioChunk::new(vec![frame])
}

pub async fn handle_control_socket(mut socket: WebSocket, handlers: Arc<ApiHandlers>) {
    let session_id = handlers.create_session();

    info!(%session_id, "WebSocket session started");

    while let Some(Ok(message)) = socket.recv().await {
        if let axum::extract::ws::Message::Binary(data) = message {
            let pcm = match decode_pcm_s16le(&data) {
                Ok(pcm) => pcm,
                Err(err) => {
                    error!(%session_id, "Failed to decode WebSocket audio: {err}");
                    continue;
                }
            };

            if let Some(chunk_pcm) = handlers.append_session_audio(&session_id, &pcm) {
                if let Some(whisper) = handlers.whisper_engine() {
                    let mut engine = whisper.lock().await;
                    let chunk = build_chunk_from_pcm(&chunk_pcm, 16_000);
                    match engine.transcribe(chunk).await {
                        Ok(transcript) => {
                            info!(%session_id, "Streaming transcript produced: {}", transcript.full_text);
                        }
                        Err(err) => error!(%session_id, "Streaming transcription failed: {err}"),
                    }
                }
            }
        }
    }

    info!(%session_id, "WebSocket session ended");
    handlers.cleanup_sessions(Duration::from_secs(300));
}
