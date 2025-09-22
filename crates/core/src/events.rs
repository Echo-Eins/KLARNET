// crates/core/src/events.rs
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// VAD events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VadEvent {
    SpeechStart {
        timestamp: DateTime<Utc>,
        confidence: f32,
    },
    SpeechFrame {
        timestamp: DateTime<Utc>,
        pcm: Arc<[f32]>,
        energy: f32,
    },
    SpeechEnd {
        timestamp: DateTime<Utc>,
        duration: Duration,
    },
    Silence {
        timestamp: DateTime<Utc>,
        duration: Duration,
    },
}

/// STT events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub confidence: f32,
    pub words: Vec<WordInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordInfo {
    pub word: String,
    pub start: f64,
    pub end: f64,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcript {
    pub id: uuid::Uuid,
    pub language: String,
    pub segments: Vec<TranscriptSegment>,
    pub full_text: String,
    pub processing_time: Duration,
}

/// NLU events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intent {
    pub name: String,
    pub confidence: f32,
    pub entities: Vec<Entity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub name: String,
    pub value: serde_json::Value,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NluResult {
    pub transcript: String,
    pub intent: Option<Intent>,
    pub wake_word_detected: bool,
    pub command_type: CommandType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommandType {
    Local(LocalCommand),
    LlmRequired(String),
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalCommand {
    pub action: String,
    pub parameters: serde_json::Map<String, serde_json::Value>,
}