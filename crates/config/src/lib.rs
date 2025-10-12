use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ack_selector::AckConfig;
use actions::ActionsConfig;
pub use klarnet_api::ApiConfig;
use klarnet_core::{AudioConfig, KlarnetResult};
use klarnet_observability::ObservabilityConfig;
use nlu::NluConfig;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tts::TtsConfig;
use vad::VadConfig;
use whisper_stt::WhisperConfig;

pub mod loader;
pub mod validator;
pub mod watcher;

pub use loader::ConfigLoader;
pub use validator::ConfigValidator;
pub use watcher::ConfigWatcher;

/// Static application section of the configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSection {
    /// Display name used in announcements and logs.
    pub assistant_name: String,
    /// Default locale for speech synthesis and recognition.
    pub language: String,
    /// Execution mode (cpu, gpu, distributed, ...).
    pub mode: String,
    /// Duration of the preroll buffer in milliseconds.
    pub pre_roll_ms: u64,
    /// Maximum length of an utterance in seconds.
    pub max_utterance_s: u64,
    /// Global log level threshold.
    pub log_level: String,
}

impl Default for AppSection {
    fn default() -> Self {
        Self {
            assistant_name: "KLARNET".to_string(),
            language: "ru".to_string(),
            mode: "gpu".to_string(),
            pre_roll_ms: 1_000,
            max_utterance_s: 120,
            log_level: "info".to_string(),
        }
    }
}

/// Metrics and Prometheus exporter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub enabled: bool,
    pub prometheus_port: u16,
    pub export_interval_s: u64,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            prometheus_port: 9090,
            export_interval_s: 10,
        }
    }
}

/// Top-level configuration structure consumed by the runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KlarnetConfig {
    pub app: AppSection,
    pub audio: AudioConfig,
    #[serde(default)]
    pub wake_word: WakeWordConfig,
    #[serde(default)]
    pub acknowledgment: AcknowledgmentConfig,
    pub vad: VadConfig,
    pub stt: WhisperConfig,
    pub nlu: NluConfig,
    pub actions: ActionsConfig,
    pub tts: TtsConfig,
    pub api: ApiConfig,
    pub metrics: MetricsConfig,
    pub observability: ObservabilityConfig,
}

impl Default for KlarnetConfig {
    fn default() -> Self {
        Self {
            app: AppSection::default(),
            audio: AudioConfig::default(),
            wake_word: WakeWordConfig::default(),
            acknowledgment: AcknowledgmentConfig::default(),
            vad: VadConfig::default(),
            stt: WhisperConfig::default(),
            nlu: NluConfig::default(),
            actions: ActionsConfig::default(),
            tts: TtsConfig::default(),
            api: ApiConfig::default(),
            metrics: MetricsConfig::default(),
            observability: ObservabilityConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WakeWordConfig {
    #[serde(default = "default_wake_word_enabled")]
    pub enabled: bool,
    #[serde(default = "default_wake_word_keyword")]
    pub keyword: String,
    #[serde(default = "default_wake_word_sensitivity")]
    pub sensitivity: f32,
    #[serde(default)]
    pub access_key: Option<String>,
}

impl Default for WakeWordConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            keyword: default_wake_word_keyword(),
            sensitivity: default_wake_word_sensitivity(),
            access_key: None,
        }
    }
}

fn default_wake_word_enabled() -> bool {
    true
}

fn default_wake_word_keyword() -> String {
    "Кларнет".to_string()
}

fn default_wake_word_sensitivity() -> f32 {
    0.7
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcknowledgmentConfig {
    #[serde(default = "default_ack_enabled")]
    pub enabled: bool,
    #[serde(flatten)]
    pub selector: AckConfig,
    #[serde(default = "default_ack_cache_enabled")]
    pub cache_enabled: bool,
}

impl Default for AcknowledgmentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            selector: AckConfig::default(),
            cache_enabled: true,
        }
    }
}

fn default_ack_enabled() -> bool {
    true
}

fn default_ack_cache_enabled() -> bool {
    true
}

/// Runtime configuration manager that exposes thread-safe access and hot reload support.
pub struct ConfigManager {
    config_path: PathBuf,
    config: Arc<RwLock<KlarnetConfig>>,
    watcher: Option<ConfigWatcher>,
    update_tx: Option<mpsc::UnboundedSender<ConfigUpdateEvent>>,
}

/// Event emitted when the configuration has been reloaded.
#[derive(Debug, Clone)]
pub struct ConfigUpdateEvent {
    pub old_config: KlarnetConfig,
    pub new_config: KlarnetConfig,
    pub changed_fields: Vec<String>,
}

impl ConfigManager {
    /// Load configuration from the provided path.
    pub fn new(config_path: impl AsRef<Path>) -> KlarnetResult<Self> {
        let path = config_path.as_ref().to_path_buf();
        let config = ConfigLoader::load_from_file(&path)?;
        ConfigValidator::validate(&config)?;

        Ok(Self {
            config_path: path,
            config: Arc::new(RwLock::new(config)),
            watcher: None,
            update_tx: None,
        })
    }

    /// Create a manager backed by default configuration values.
    pub fn with_defaults() -> Self {
        Self {
            config_path: PathBuf::from("config/klarnet.toml"),
            config: Arc::new(RwLock::new(KlarnetConfig::default())),
            watcher: None,
            update_tx: None,
        }
    }

    /// Enable hot reloading. The returned receiver yields events when the configuration changes.
    pub fn enable_hot_reload(
        &mut self,
    ) -> KlarnetResult<mpsc::UnboundedReceiver<ConfigUpdateEvent>> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.update_tx = Some(tx.clone());

        let watcher = ConfigWatcher::start(self.config_path.clone(), Arc::clone(&self.config), tx)?;
        self.watcher = Some(watcher);

        Ok(rx)
    }

    /// Retrieve a copy of the current configuration.
    pub fn get(&self) -> KlarnetConfig {
        self.config.read().clone()
    }

    /// Apply a mutating function to the configuration and persist it on disk.
    pub fn update<F>(&self, mutator: F) -> KlarnetResult<()>
    where
        F: FnOnce(&mut KlarnetConfig),
    {
        {
            let mut guard = self.config.write();
            mutator(&mut guard);
            ConfigValidator::validate(&guard)?;
            ConfigLoader::save_to_file(&self.config_path, &guard)?;
        }

        Ok(())
    }

    /// Force a reload from disk and emit an update event if there are changes.
    pub fn reload(&self) -> KlarnetResult<Option<ConfigUpdateEvent>> {
        let new_config = ConfigLoader::load_from_file(&self.config_path)?;
        ConfigValidator::validate(&new_config)?;

        let mut guard = self.config.write();
        let old_config = guard.clone();

        let changed_fields = detect_changes(&old_config, &new_config);
        if changed_fields.is_empty() {
            return Ok(None);
        }
        *guard = new_config.clone();
        drop(guard);

        let event = ConfigUpdateEvent {
            old_config,
            new_config,
            changed_fields,
        };

        if let Some(tx) = &self.update_tx {
            let _ = tx.send(event.clone());
        }

        Ok(Some(event))
    }
}

fn detect_changes(old: &KlarnetConfig, new: &KlarnetConfig) -> Vec<String> {
    let old_value = serde_json::to_value(old).unwrap_or_default();
    let new_value = serde_json::to_value(new).unwrap_or_default();
    let mut changes = BTreeSet::new();
    diff_recursive("", &old_value, &new_value, &mut changes);
    changes.into_iter().collect()
}

fn diff_recursive(
    prefix: &str,
    old: &serde_json::Value,
    new: &serde_json::Value,
    acc: &mut BTreeSet<String>,
) {
    if old == new {
        return;
    }

    match (old, new) {
        (serde_json::Value::Object(old_map), serde_json::Value::Object(new_map)) => {
            let keys: BTreeSet<_> = old_map.keys().chain(new_map.keys()).cloned().collect();

            for key in keys {
                let new_prefix = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{}.{}", prefix, key)
                };

                diff_recursive(
                    &new_prefix,
                    old_map.get(&key).unwrap_or(&serde_json::Value::Null),
                    new_map.get(&key).unwrap_or(&serde_json::Value::Null),
                    acc,
                );
            }
        }
        _ => {
            acc.insert(prefix.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_reports_field_changes() {
        let mut old = KlarnetConfig::default();
        let mut new = KlarnetConfig::default();
        new.audio.sample_rate = 48_000;
        new.api.port = 4000;
        new.app.assistant_name = "Jarvis".to_string();

        let changes = detect_changes(&old, &new);
        assert!(changes.contains(&"audio.sample_rate".to_string()));
        assert!(changes.contains(&"api.port".to_string()));
        assert!(changes.contains(&"app.assistant_name".to_string()));

        old.nlu.wake_words.push("кларнет".to_string());
        new.nlu.wake_words.push("джарвис".to_string());
        let changes = detect_changes(&old, &new);
        assert!(changes.iter().any(|c| c.starts_with("nlu.wake_words")));
    }
}
