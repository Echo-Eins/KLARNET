use std::path::PathBuf;
use std::time::Duration;

use klarnet_core::{resolve_python_path, KlarnetError, KlarnetResult};
use serde::{Deserialize, Serialize};

pub const SUPPORTED_LANGUAGES: &[&str] = &[
    "af", "am", "ar", "as", "az", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy", "da", "de",
    "el", "en", "es", "et", "fa", "fi", "fr", "fy", "ga", "gd", "gl", "gu", "haw", "he", "hi",
    "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "la", "lb",
    "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn",
    "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq",
    "sr", "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "uz", "vi", "yi", "yo", "zh",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperModelConfig {
    pub model_path: PathBuf,
    #[serde(default = "WhisperModelConfig::default_device")]
    pub device: String,
    #[serde(default = "WhisperModelConfig::default_compute_type")]
    pub compute_type: String,
    #[serde(default)]
    pub cache_dir: Option<PathBuf>,
}

impl WhisperModelConfig {
    fn default_device() -> String {
        "cpu".to_string()
    }

    fn default_compute_type() -> String {
        "int8".to_string()
    }
}

impl Default for WhisperModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/whisper-medium"),
            device: Self::default_device(),
            compute_type: Self::default_compute_type(),
            cache_dir: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperPythonConfig {
    #[serde(default = "WhisperPythonConfig::default_executable")]
    pub executable: PathBuf,
    #[serde(default = "WhisperPythonConfig::default_script")]
    pub script: PathBuf,
    #[serde(default)]
    pub extra_args: Vec<String>,
    #[serde(default)]
    pub env: Vec<(String, String)>,
}

impl WhisperPythonConfig {
    fn default_executable() -> PathBuf {
        resolve_python_path()
    }

    fn default_script() -> PathBuf {
        PathBuf::from("scripts/whisper_server.py")
    }
}

impl Default for WhisperPythonConfig {
    fn default() -> Self {
        Self {
            executable: Self::default_executable(),
            script: Self::default_script(),
            extra_args: Vec::new(),
            env: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WhisperBackendConfig {
    Python(WhisperPythonConfig),
}

impl Default for WhisperBackendConfig {
    fn default() -> Self {
        Self::Python(WhisperPythonConfig::default())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperConfig {
    #[serde(default = "WhisperConfig::default_language")]
    pub language: String,
    #[serde(default)]
    pub model: WhisperModelConfig,
    #[serde(default)]
    pub backend: WhisperBackendConfig,
    #[serde(default = "WhisperConfig::default_request_timeout_ms")]
    pub request_timeout_ms: u64,
    #[serde(default = "WhisperConfig::default_initialization_timeout_ms")]
    pub initialization_timeout_ms: u64,
    #[serde(default = "WhisperConfig::default_retry_attempts")]
    pub retry_attempts: usize,
    #[serde(default = "WhisperConfig::default_retry_backoff_ms")]
    pub retry_backoff_ms: u64,
}

impl WhisperConfig {
    fn default_language() -> String {
        "ru".to_string()
    }

    fn default_request_timeout_ms() -> u64 {
        30_000
    }

    fn default_initialization_timeout_ms() -> u64 {
        120_000
    }

    fn default_retry_attempts() -> usize {
        2
    }

    fn default_retry_backoff_ms() -> u64 {
        500
    }

    pub fn request_timeout(&self) -> Duration {
        Duration::from_millis(self.request_timeout_ms)
    }

    pub fn initialization_timeout(&self) -> Duration {
        Duration::from_millis(self.initialization_timeout_ms)
    }

    pub fn retry_backoff(&self) -> Duration {
        Duration::from_millis(self.retry_backoff_ms)
    }

    pub fn validate(&self) -> KlarnetResult<()> {
        if self.language.trim().is_empty() {
            return Err(KlarnetError::Config(
                "Whisper language must not be empty".to_string(),
            ));
        }

        if !SUPPORTED_LANGUAGES
            .iter()
            .any(|lang| lang.eq_ignore_ascii_case(&self.language))
        {
            return Err(KlarnetError::Config(format!(
                "Unsupported Whisper language: {}",
                self.language
            )));
        }

        Ok(())
    }
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            language: Self::default_language(),
            model: WhisperModelConfig::default(),
            backend: WhisperBackendConfig::default(),
            request_timeout_ms: Self::default_request_timeout_ms(),
            initialization_timeout_ms: Self::default_initialization_timeout_ms(),
            retry_attempts: Self::default_retry_attempts(),
            retry_backoff_ms: Self::default_retry_backoff_ms(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct WhisperMetrics {
    pub total_processed: u64,
    pub total_errors: u64,
    pub total_timeouts: u64,
    pub total_retries: u64,
    pub total_restarts: u64,
    pub cumulative_processing_time: Duration,
    pub last_processing_time: Option<Duration>,
}

impl WhisperMetrics {
    pub fn average_processing_time(&self) -> Option<Duration> {
        if self.total_processed == 0 {
            return None;
        }

        Some(self.cumulative_processing_time / self.total_processed as u32)
    }
}
