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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AppConfig {
    pub language: String,
    pub mode: String,
    pub pre_roll_ms: u64,
    pub max_utterance_s: u64,
    pub log_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub buffer_size: usize,
    pub device: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VadConfig {
    pub mode: String,
    pub aggressiveness: u8,
    pub frame_duration_ms: usize,
    pub min_speech_duration_ms: usize,
    pub min_silence_duration_ms: usize,
    pub energy_threshold: f32,
    pub speech_pad_ms: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NluConfig {
    pub mode: String,
    pub wake_words: Vec<String>,
    pub confidence_threshold: f32,
    pub local: Option<LocalNluConfig>,
    pub llm: Option<LlmNluConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LocalNluConfig {
    pub patterns_file: PathBuf,
    pub entities_file: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActionsConfig {
    pub enabled_modules: Vec<String>,
    pub scripts_dir: Option<PathBuf>,
    pub smart_home: Option<SmartHomeConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SmartHomeConfig {
    pub api_url: String,
    pub api_token_env: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TtsCacheConfig {
    pub enabled: bool,
    pub directory: PathBuf,
    pub max_entries: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TtsMonitoringConfig {
    pub enabled: bool,
    pub min_rms: f32,
    pub max_latency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TtsRuntimeConfig {
    pub python_path: PathBuf,
    pub silero_script: PathBuf,
    pub piper_binary: PathBuf,
    pub request_timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TtsConfig {
    pub enabled: bool,
    pub engine: String,
    pub language: String,
    pub model: String,
    pub speaker: String,
    pub sample_rate: u32,
    pub speed: f32,
    pub device: Option<String>,
    pub cache: TtsCacheConfig,
    pub monitoring: TtsMonitoringConfig,
    pub runtime: TtsRuntimeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiConfig {
    pub enabled: bool,
    pub host: String,
    pub port: u16,
    pub cors_origins: Vec<String>,
    pub auth: Option<AuthConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AuthConfig {
    pub enabled: bool,
    pub jwt_secret_env: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricsConfig {
    pub enabled: bool,
    pub prometheus_port: u16,
    pub export_interval_s: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
                    model: "x-ai/grok-4-fast:free".to_string(),
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
                language: "ru".to_string(),
                model: "v3_1_ru".to_string(),
                speaker: "xenia".to_string(),
                sample_rate: 48000,
                speed: 1.0,
                device: Some("cpu".to_string()),
                cache: TtsCacheConfig {
                    enabled: true,
                    directory: PathBuf::from("cache/tts"),
                    max_entries: 128,
                },
                monitoring: TtsMonitoringConfig {
                    enabled: true,
                    min_rms: 0.01,
                    max_latency_ms: 5_000,
                },
                runtime: TtsRuntimeConfig {
                    python_path: PathBuf::from("python3"),
                    silero_script: PathBuf::from("scripts/silero_tts.py"),
                    piper_binary: PathBuf::from("piper"),
                    request_timeout_ms: 15_000,
                },
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

        macro_rules! compare_field {
            ($old:expr, $new:expr, $name:expr) => {
                if $old != $new {
                    changes.push($name.to_string());
                }
            };
        }

        compare_field!(old.app.language, new.app.language, "app.language");
        compare_field!(old.app.mode, new.app.mode, "app.mode");
        compare_field!(
            old.app.pre_roll_ms,
            new.app.pre_roll_ms,
            "app.pre_roll_ms"
        );
        compare_field!(
            old.app.max_utterance_s,
            new.app.max_utterance_s,
            "app.max_utterance_s"
        );
        compare_field!(old.app.log_level, new.app.log_level, "app.log_level");

        compare_field!(
            old.audio.sample_rate,
            new.audio.sample_rate,
            "audio.sample_rate"
        );
        compare_field!(old.audio.channels, new.audio.channels, "audio.channels");
        compare_field!(
            old.audio.bits_per_sample,
            new.audio.bits_per_sample,
            "audio.bits_per_sample"
        );
        compare_field!(
            old.audio.buffer_size,
            new.audio.buffer_size,
            "audio.buffer_size"
        );
        compare_field!(old.audio.device, new.audio.device, "audio.device");

        compare_field!(old.vad.mode, new.vad.mode, "vad.mode");
        compare_field!(
            old.vad.aggressiveness,
            new.vad.aggressiveness,
            "vad.aggressiveness"
        );
        compare_field!(
            old.vad.frame_duration_ms,
            new.vad.frame_duration_ms,
            "vad.frame_duration_ms"
        );
        compare_field!(
            old.vad.min_speech_duration_ms,
            new.vad.min_speech_duration_ms,
            "vad.min_speech_duration_ms"
        );
        compare_field!(
            old.vad.min_silence_duration_ms,
            new.vad.min_silence_duration_ms,
            "vad.min_silence_duration_ms"
        );
        compare_field!(
            old.vad.energy_threshold,
            new.vad.energy_threshold,
            "vad.energy_threshold"
        );
        compare_field!(
            old.vad.speech_pad_ms,
            new.vad.speech_pad_ms,
            "vad.speech_pad_ms"
        );

        compare_field!(
            old.stt.model_path,
            new.stt.model_path,
            "stt.model_path"
        );
        compare_field!(old.stt.model_size, new.stt.model_size, "stt.model_size");
        compare_field!(
            old.stt.compute_type,
            new.stt.compute_type,
            "stt.compute_type"
        );
        compare_field!(old.stt.language, new.stt.language, "stt.language");
        compare_field!(old.stt.beam_size, new.stt.beam_size, "stt.beam_size");
        compare_field!(old.stt.vad_filter, new.stt.vad_filter, "stt.vad_filter");
        compare_field!(
            old.stt.word_timestamps,
            new.stt.word_timestamps,
            "stt.word_timestamps"
        );
        compare_field!(old.stt.device, new.stt.device, "stt.device");
        compare_field!(
            old.stt.device_index,
            new.stt.device_index,
            "stt.device_index"
        );
        compare_field!(old.stt.num_workers, new.stt.num_workers, "stt.num_workers");
        compare_field!(old.stt.batch_size, new.stt.batch_size, "stt.batch_size");
        compare_field!(
            old.stt.max_segment_length,
            new.stt.max_segment_length,
            "stt.max_segment_length"
        );

        compare_field!(old.nlu.mode, new.nlu.mode, "nlu.mode");
        compare_field!(old.nlu.wake_words, new.nlu.wake_words, "nlu.wake_words");
        compare_field!(
            old.nlu.confidence_threshold,
            new.nlu.confidence_threshold,
            "nlu.confidence_threshold"
        );

        match (&old.nlu.local, &new.nlu.local) {
            (Some(old_local), Some(new_local)) => {
                compare_field!(
                    old_local.patterns_file,
                    new_local.patterns_file,
                    "nlu.local.patterns_file"
                );
                compare_field!(
                    old_local.entities_file,
                    new_local.entities_file,
                    "nlu.local.entities_file"
                );
            }
            (None, None) => {}
            _ => changes.push("nlu.local".to_string()),
        }

        match (&old.nlu.llm, &new.nlu.llm) {
            (Some(old_llm), Some(new_llm)) => {
                compare_field!(old_llm.provider, new_llm.provider, "nlu.llm.provider");
                compare_field!(old_llm.model, new_llm.model, "nlu.llm.model");
                compare_field!(
                    old_llm.api_key_env,
                    new_llm.api_key_env,
                    "nlu.llm.api_key_env"
                );
                compare_field!(
                    old_llm.max_tokens,
                    new_llm.max_tokens,
                    "nlu.llm.max_tokens"
                );
                compare_field!(
                    old_llm.temperature,
                    new_llm.temperature,
                    "nlu.llm.temperature"
                );
                compare_field!(old_llm.timeout_s, new_llm.timeout_s, "nlu.llm.timeout_s");
                compare_field!(old_llm.base_url, new_llm.base_url, "nlu.llm.base_url");
                compare_field!(old_llm.top_p, new_llm.top_p, "nlu.llm.top_p");
                compare_field!(
                    old_llm.retry_attempts,
                    new_llm.retry_attempts,
                    "nlu.llm.retry_attempts"
                );
                compare_field!(
                    old_llm.cache_enabled,
                    new_llm.cache_enabled,
                    "nlu.llm.cache_enabled"
                );
                compare_field!(old_llm.cache_ttl_s, new_llm.cache_ttl_s, "nlu.llm.cache_ttl_s");
                compare_field!(
                    old_llm.max_concurrent_requests,
                    new_llm.max_concurrent_requests,
                    "nlu.llm.max_concurrent_requests"
                );
                compare_field!(
                    old_llm.min_request_interval_ms,
                    new_llm.min_request_interval_ms,
                    "nlu.llm.min_request_interval_ms"
                );
            }
            (None, None) => {}
            _ => changes.push("nlu.llm".to_string()),
        }
        compare_field!(
            old.actions.enabled_modules,
            new.actions.enabled_modules,
            "actions.enabled_modules"
        );
        compare_field!(old.actions.scripts_dir, new.actions.scripts_dir, "actions.scripts_dir");

        match (&old.actions.smart_home, &new.actions.smart_home) {
            (Some(old_sh), Some(new_sh)) => {
                compare_field!(
                    old_sh.api_url,
                    new_sh.api_url,
                    "actions.smart_home.api_url"
                );
                compare_field!(
                    old_sh.api_token_env,
                    new_sh.api_token_env,
                    "actions.smart_home.api_token_env"
                );
            }
            (None, None) => {}
            _ => changes.push("actions.smart_home".to_string()),
        }
        compare_field!(old.tts.enabled, new.tts.enabled, "tts.enabled");
        compare_field!(old.tts.engine, new.tts.engine, "tts.engine");
        compare_field!(old.tts.language, new.tts.language, "tts.language");
        compare_field!(old.tts.model, new.tts.model, "tts.model");
        compare_field!(old.tts.speaker, new.tts.speaker, "tts.speaker");
        compare_field!(old.tts.sample_rate, new.tts.sample_rate, "tts.sample_rate");
        compare_field!(old.tts.speed, new.tts.speed, "tts.speed");
        compare_field!(old.tts.device, new.tts.device, "tts.device");

        compare_field!(old.tts.cache.enabled, new.tts.cache.enabled, "tts.cache.enabled");
        compare_field!(
            old.tts.cache.directory,
            new.tts.cache.directory,
            "tts.cache.directory"
        );
        compare_field!(
            old.tts.cache.max_entries,
            new.tts.cache.max_entries,
            "tts.cache.max_entries"
        );

        compare_field!(
            old.tts.monitoring.enabled,
            new.tts.monitoring.enabled,
            "tts.monitoring.enabled"
        );
        compare_field!(
            old.tts.monitoring.min_rms,
            new.tts.monitoring.min_rms,
            "tts.monitoring.min_rms"
        );
        compare_field!(
            old.tts.monitoring.max_latency_ms,
            new.tts.monitoring.max_latency_ms,
            "tts.monitoring.max_latency_ms"
        );

        compare_field!(
            old.tts.runtime.python_path,
            new.tts.runtime.python_path,
            "tts.runtime.python_path"
        );
        compare_field!(
            old.tts.runtime.silero_script,
            new.tts.runtime.silero_script,
            "tts.runtime.silero_script"
        );
        compare_field!(
            old.tts.runtime.piper_binary,
            new.tts.runtime.piper_binary,
            "tts.runtime.piper_binary"
        );
        compare_field!(
            old.tts.runtime.request_timeout_ms,
            new.tts.runtime.request_timeout_ms,
            "tts.runtime.request_timeout_ms"
        );

        compare_field!(old.api.enabled, new.api.enabled, "api.enabled");
        compare_field!(old.api.host, new.api.host, "api.host");
        compare_field!(old.api.port, new.api.port, "api.port");
        compare_field!(old.api.cors_origins, new.api.cors_origins, "api.cors_origins");

        match (&old.api.auth, &new.api.auth) {
            (Some(old_auth), Some(new_auth)) => {
                compare_field!(old_auth.enabled, new_auth.enabled, "api.auth.enabled");
                compare_field!(
                    old_auth.jwt_secret_env,
                    new_auth.jwt_secret_env,
                    "api.auth.jwt_secret_env"
                );
            }
            (None, None) => {}
            _ => changes.push("api.auth".to_string()),
        }
        compare_field!(old.metrics.enabled, new.metrics.enabled, "metrics.enabled");
        compare_field!(
            old.metrics.prometheus_port,
            new.metrics.prometheus_port,
            "metrics.prometheus_port"
        );
        compare_field!(
            old.metrics.export_interval_s,
            new.metrics.export_interval_s,
            "metrics.export_interval_s"
        );

        compare_field!(
            old.observability.traces_enabled,
            new.observability.traces_enabled,
            "observability.traces_enabled"
        );
        compare_field!(
            old.observability.traces_endpoint,
            new.observability.traces_endpoint,
            "observability.traces_endpoint"
        );
        compare_field!(
            old.observability.service_name,
            new.observability.service_name,
            "observability.service_name"
        );

        changes
    }
}
