// crates/config/src/lib.rs

use klarnet_core::{KlarnetError, KlarnetResult};
use notify::{Event, RecursiveMode, Watcher};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

pub mod validator;
pub mod loader;
pub mod watcher;

use validator::ConfigValidator;
use loader::ConfigLoader;

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KlarnetConfig {
    pub app: AppConfig,
    pub audio: AudioConfig,
    pub vad: VadConfig,
    pub stt: SttConfig,
    pub nlu: NluConfig,
    pub actions: ActionsConfig,
    pub tts: TtsConfig,
    pub api: ApiConfig,
    pub metrics: MetricsConfig,
    pub observability: ObservabilityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub language: String,
    pub mode: String,
    pub pre_roll_ms: u64,
    pub max_utterance_s: u64,
    pub log_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub buffer_size: usize,
    pub device: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadConfig {
    pub mode: String,
    pub aggressiveness: u8,
    pub frame_duration_ms: usize,
    pub min_speech_duration_ms: usize,
    pub min_silence_duration_ms: usize,
    pub energy_threshold: f32,
    pub speech_pad_ms: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttConfig {
    pub model_path: PathBuf,
    pub model_size: String,
    pub compute_type: String,
    pub language: String,
    pub beam_size: usize,
    pub vad_filter: bool,
    pub word_timestamps: bool,
    pub device: String,
    pub device_index: Option<usize>,
    pub num_workers: usize,
    pub batch_size: usize,
    pub max_segment_length: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NluConfig {
    pub mode: String,
    pub wake_words: Vec<String>,
    pub confidence_threshold: f32,
    pub local: Option<LocalNluConfig>,
    pub llm: Option<LlmNluConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalNluConfig {
    pub patterns_file: PathBuf,
    pub entities_file: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmNluConfig {
    pub provider: String,
    pub model: String,
    pub api_key_env: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub timeout_s: u64,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default = "default_llm_top_p")]
    pub top_p: f32,
    #[serde(default = "default_llm_retry_attempts")]
    pub retry_attempts: u32,
    #[serde(default = "default_llm_cache_enabled")]
    pub cache_enabled: bool,
    #[serde(default = "default_llm_cache_ttl_s")]
    pub cache_ttl_s: u64,
    #[serde(default = "default_llm_max_concurrent_requests")]
    pub max_concurrent_requests: usize,
    #[serde(default = "default_llm_min_request_interval_ms")]
    pub min_request_interval_ms: u64,
}

const fn default_llm_top_p() -> f32 {
    0.95
}

const fn default_llm_retry_attempts() -> u32 {
    3
}

const fn default_llm_cache_enabled() -> bool {
    true
}

const fn default_llm_cache_ttl_s() -> u64 {
    3600
}

const fn default_llm_max_concurrent_requests() -> usize {
    1
}

const fn default_llm_min_request_interval_ms() -> u64 {
    0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionsConfig {
    pub enabled_modules: Vec<String>,
    pub scripts_dir: Option<PathBuf>,
    pub smart_home: Option<SmartHomeConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartHomeConfig {
    pub api_url: String,
    pub api_token_env: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsConfig {
    pub enabled: bool,
    pub engine: String,
    pub model: String,
    pub speaker: String,
    pub sample_rate: u32,
    pub speed: f32,
    pub device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub enabled: bool,
    pub host: String,
    pub port: u16,
    pub cors_origins: Vec<String>,
    pub auth: Option<AuthConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    pub enabled: bool,
    pub jwt_secret_env: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub enabled: bool,
    pub prometheus_port: u16,
    pub export_interval_s: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    pub traces_enabled: bool,
    pub traces_endpoint: Option<String>,
    pub service_name: String,
}

impl Default for KlarnetConfig {
    fn default() -> Self {
        Self {
            app: AppConfig {
                language: "ru".to_string(),
                mode: "gpu".to_string(),
                pre_roll_ms: 1000,
                max_utterance_s: 120,
                log_level: "info".to_string(),
            },
            audio: AudioConfig {
                sample_rate: 16000,
                channels: 1,
                bits_per_sample: 16,
                buffer_size: 1024,
                device: None,
            },
            vad: VadConfig {
                mode: "webrtc".to_string(),
                aggressiveness: 2,
                frame_duration_ms: 30,
                min_speech_duration_ms: 200,
                min_silence_duration_ms: 500,
                energy_threshold: 0.01,
                speech_pad_ms: 300,
            },
            stt: SttConfig {
                model_path: PathBuf::from("models/faster-whisper-medium"),
                model_size: "medium".to_string(),
                compute_type: "int8_float16".to_string(),
                language: "ru".to_string(),
                beam_size: 5,
                vad_filter: true,
                word_timestamps: true,
                device: "cuda".to_string(),
                device_index: Some(0),
                num_workers: 1,
                batch_size: 1,
                max_segment_length: 30.0,
            },
            nlu: NluConfig {
                mode: "hybrid".to_string(),
                wake_words: vec!["джарвис".to_string(), "ассистент".to_string()],
                confidence_threshold: 0.7,
                local: Some(LocalNluConfig {
                    patterns_file: PathBuf::from("config/patterns.yaml"),
                    entities_file: PathBuf::from("config/entities.yaml"),
                }),
                llm: Some(LlmNluConfig {
                    provider: "openrouter".to_string(),
                    model: "deepseek/deepseek-chat".to_string(),
                    api_key_env: "OPENROUTER_API_KEY".to_string(),
                    max_tokens: 500,
                    temperature: 0.3,
                    timeout_s: 5,
                    base_url: None,
                    top_p: default_llm_top_p(),
                    retry_attempts: default_llm_retry_attempts(),
                    cache_enabled: default_llm_cache_enabled(),
                    cache_ttl_s: default_llm_cache_ttl_s(),
                    max_concurrent_requests: default_llm_max_concurrent_requests(),
                    min_request_interval_ms: default_llm_min_request_interval_ms(),
                }),
            },
            actions: ActionsConfig {
                enabled_modules: vec!["system".to_string(), "smart_home".to_string()],
                scripts_dir: Some(PathBuf::from("scripts/actions")),
                smart_home: Some(SmartHomeConfig {
                    api_url: "http://homeassistant.local:8123".to_string(),
                    api_token_env: "HASS_TOKEN".to_string(),
                }),
            },
            tts: TtsConfig {
                enabled: true,
                engine: "silero".to_string(),
                model: "v3_1_ru".to_string(),
                speaker: "xenia".to_string(),
                sample_rate: 48000,
                speed: 1.0,
                device: "cpu".to_string(),
            },
            api: ApiConfig {
                enabled: true,
                host: "0.0.0.0".to_string(),
                port: 3000,
                cors_origins: vec!["*".to_string()],
                auth: None,
            },
            metrics: MetricsConfig {
                enabled: true,
                prometheus_port: 9090,
                export_interval_s: 10,
            },
            observability: ObservabilityConfig {
                traces_enabled: true,
                traces_endpoint: Some("http://localhost:4317".to_string()),
                service_name: "klarnet".to_string(),
            },
        }
    }
}

/// Configuration manager with hot reload support
pub struct ConfigManager {
    config_path: PathBuf,
    config: Arc<RwLock<KlarnetConfig>>,
    watcher: Option<notify::RecommendedWatcher>,
    update_tx: Option<mpsc::UnboundedSender<ConfigUpdateEvent>>,
}

#[derive(Debug, Clone)]
pub struct ConfigUpdateEvent {
    pub old_config: KlarnetConfig,
    pub new_config: KlarnetConfig,
    pub changed_fields: Vec<String>,
}

impl ConfigManager {
    pub fn new(config_path: impl AsRef<Path>) -> KlarnetResult<Self> {
        let config_path = config_path.as_ref().to_path_buf();
        let config = ConfigLoader::load_from_file(&config_path)?;

        // Validate configuration
        ConfigValidator::validate(&config)?;

        Ok(Self {
            config_path,
            config: Arc::new(RwLock::new(config)),
            watcher: None,
            update_tx: None,
        })
    }

    pub fn with_defaults() -> Self {
        Self {
            config_path: PathBuf::from("config/klarnet.toml"),
            config: Arc::new(RwLock::new(KlarnetConfig::default())),
            watcher: None,
            update_tx: None,
        }
    }

    pub fn enable_hot_reload(&mut self) -> KlarnetResult<mpsc::UnboundedReceiver<ConfigUpdateEvent>> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.update_tx = Some(tx.clone());

        let config_path = self.config_path.clone();
        let config_arc = self.config.clone();

        let mut watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
            match res {
                Ok(event) => {
                    if event.kind.is_modify() {
                        info!("Configuration file changed, reloading...");

                        match ConfigLoader::load_from_file(&config_path) {
                            Ok(new_config) => {
                                if let Err(e) = ConfigValidator::validate(&new_config) {
                                    error!("Invalid configuration: {}", e);
                                    return;
                                }

                                let old_config = config_arc.read().clone();
                                let changed_fields = Self::detect_changes(&old_config, &new_config);

                                if !changed_fields.is_empty() {
                                    info!("Configuration changes detected: {:?}", changed_fields);

                                    *config_arc.write() = new_config.clone();

                                    let _ = tx.send(ConfigUpdateEvent {
                                        old_config,
                                        new_config,
                                        changed_fields,
                                    });
                                }
                            }
                            Err(e) => {
                                error!("Failed to reload configuration: {}", e);
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Configuration watcher error: {}", e);
                }
            }
        }).map_err(|e| KlarnetError::Config(format!("Failed to create watcher: {}", e)))?;

        watcher.watch(&self.config_path, RecursiveMode::NonRecursive)
            .map_err(|e| KlarnetError::Config(format!("Failed to watch config: {}", e)))?;

        self.watcher = Some(watcher);

        Ok(rx)
    }

    pub fn get(&self) -> KlarnetConfig {
        self.config.read().clone()
    }

    pub fn update<F>(&self, f: F) -> KlarnetResult<()>
    where
        F: FnOnce(&mut KlarnetConfig),
    {
        let mut config = self.config.write();
        f(&mut config);

        // Validate after update
        ConfigValidator::validate(&config)?;

        Ok(())
    }

    pub fn save(&self) -> KlarnetResult<()> {
        let config = self.config.read();
        ConfigLoader::save_to_file(&self.config_path, &config)?;
        Ok(())
    }

    fn detect_changes(old: &KlarnetConfig, new: &KlarnetConfig) -> Vec<String> {
        let mut changes = Vec::new();

        // Compare each field (simplified)
        if old.app.language != new.app.language {
            changes.push("app.language".to_string());
        }
        if old.vad.aggressiveness != new.vad.aggressiveness {
            changes.push("vad.aggressiveness".to_string());
        }
        if old.stt.beam_size != new.stt.beam_size {
            changes.push("stt.beam_size".to_string());
        }
        // ... compare other fields

        changes
    }
}
