// crates/api/src/handlers.rs

use axum::extract::ws::{Message, WebSocket};
use bytes::Bytes;
use futures::{sink::SinkExt, stream::StreamExt};
use klarnet_core::{AudioChunk, KlarnetResult, Transcript};
use klarnet_nlu::NluEngine;
use klarnet_whisper_stt::WhisperEngine;
use parking_lot::RwLock;
use serde_json::json;
use std::sync::Arc;
use tracing::{debug, error, info};

pub struct ApiHandlers {
    whisper: Option<Arc<WhisperEngine>>,
    nlu: Option<Arc<NluEngine>>,
    metrics: Arc<klarnet_observability::MetricsCollector>,
    active_sessions: Arc<RwLock<SessionManager>>,
}

impl ApiHandlers {
    pub fn new(metrics: Arc<klarnet_observability::MetricsCollector>) -> Self {
        Self {
            whisper: None,
            nlu: None,
            metrics,
            active_sessions: Arc::new(RwLock::new(SessionManager::new())),
        }
    }

    pub async fn with_engines(
        whisper: Arc<WhisperEngine>,
        nlu: Arc<NluEngine>,
        metrics: Arc<klarnet_observability::MetricsCollector>,
    ) -> Self {
        Self {
            whisper: Some(whisper),
            nlu: Some(nlu),
            metrics,
            active_sessions: Arc::new(RwLock::new(SessionManager::new())),
        }
    }

    pub async fn transcribe_file(&self, audio_data: Bytes) -> KlarnetResult<Transcript> {
        let whisper = self.whisper.as_ref()
            .ok_or_else(|| klarnet_core::KlarnetError::Stt("STT engine not initialized".to_string()))?;

        // Convert bytes to PCM float32
        let pcm = self.decode_audio(&audio_data)?;

        // Create audio chunk
        let chunk = AudioChunk::from_pcm(&pcm, 16000);

        // Transcribe
        let transcript = whisper.transcribe(chunk).await?;

        // Update metrics
        self.metrics.increment(klarnet_observability::metrics::MetricType::TranscriptionsCompleted);

        Ok(transcript)
    }

    pub async fn interpret_text(&self, text: String) -> KlarnetResult<serde_json::Value> {
        let nlu = self.nlu.as_ref()
            .ok_or_else(|| klarnet_core::KlarnetError::Nlu("NLU engine not initialized".to_string()))?;

        // Create dummy transcript for NLU
        let transcript = Transcript {
            id: uuid::Uuid::new_v4(),
            language: "ru".to_string(),
            segments: vec![],
            full_text: text,
            processing_time: std::time::Duration::from_millis(0),
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
        // Process with LLM if available
        let response = if let Some(nlu) = &self.nlu {
            let transcript = Transcript {
                id: uuid::Uuid::new_v4(),
                language: "ru".to_string(),
                segments: vec![],
                full_text: request.message.clone(),
                processing_time: std::time::Duration::from_millis(0),
            };

            let result = nlu.process(&transcript).await?;

            super::ChatResponse {
                response: format!("Обработано: {}", result.transcript),
                action: None,
            }
        } else {
            super::ChatResponse {
                response: "Chat processing not available".to_string(),
                action: None,
            }
        };

        Ok(response)
    }

    pub async fn get_metrics(&self) -> String {
        self.metrics.get_prometheus_metrics()
    }

    fn decode_audio(&self, data: &[u8]) -> KlarnetResult<Vec<f32>> {
        // Simplified audio decoding - in production, use proper audio decoder
        let pcm: Vec<f32> = data.chunks_exact(2)
            .map(|chunk| {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                sample as f32 / 32768.0
            })
            .collect();

        Ok(pcm)
    }
}

struct SessionManager {
    sessions: std::collections::HashMap<uuid::Uuid, Session>,
}

struct Session {
    id: uuid::Uuid,
    created_at: std::time::Instant,
    last_activity: std::time::Instant,
    audio_buffer: Vec<f32>,
}

impl SessionManager {
    fn new() -> Self {
        Self {
            sessions: std::collections::HashMap::new(),
        }
    }

    fn create_session(&mut self) -> uuid::Uuid {
        let id = uuid::Uuid::new_v4();
        let session = Session {
            id,
            created_at: std::time::Instant::now(),
            last_activity: std::time::Instant::now(),
            audio_buffer: Vec::new(),
        };

        self.sessions.insert(id, session);
        id
    }

    fn get_session_mut(&mut self, id: &uuid::Uuid) -> Option<&mut Session> {
        if let Some(session) = self.sessions.get_mut(id) {
            session.last_activity = std::time::Instant::now();
            Some(session)
        } else {
            None
        }
    }

    fn cleanup_old_sessions(&mut self, max_age: std::time::Duration) {
        let now = std::time::Instant::now();
        self.sessions.retain(|_, session| {
            now - session.last_activity < max_age
        });
    }
}