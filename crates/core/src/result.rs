// crates/core/src/result.rs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum KlarnetError {
    #[error("Audio error: {0}")]
    Audio(String),

    #[error("VAD error: {0}")]
    Vad(String),

    #[error("STT error: {0}")]
    Stt(String),

    #[error("NLU error: {0}")]
    Nlu(String),

    #[error("Action error: {0}")]
    Action(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

pub type KlarnetResult<T> = Result<T, KlarnetError>;
