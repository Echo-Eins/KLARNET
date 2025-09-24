// crates/tts/src/lib.rs

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use klarnet_core::{KlarnetError, KlarnetResult};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

pub mod piper;
pub mod player;
pub mod silero;
use player::AudioPlayer;

/// Cache configuration for synthesized audio snippets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsCacheConfig {
    pub enabled: bool,
    pub directory: PathBuf,
    pub max_entries: usize,
}

impl Default for TtsCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            directory: PathBuf::from("cache/tts"),
            max_entries: 128,
        }
    }
}

/// Monitoring thresholds for synthesized audio quality and latency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsMonitoringConfig {
    pub enabled: bool,
    /// Minimum acceptable root-mean-square signal level (0.0 - 1.0).
    pub min_rms: f32,
    /// Maximum acceptable backend latency in milliseconds.
    pub max_latency_ms: u64,
}

impl Default for TtsMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_rms: 0.01,
            max_latency_ms: 5_000,
        }
    }
}

/// Runtime configuration required by individual TTS engines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsRuntimeConfig {
    pub python_path: PathBuf,
    pub silero_script: PathBuf,
    pub piper_binary: PathBuf,
    pub request_timeout_ms: u64,
}

impl Default for TtsRuntimeConfig {
    fn default() -> Self {
        Self {
            python_path: PathBuf::from("python3"),
            silero_script: PathBuf::from("scripts/silero_tts.py"),
            piper_binary: PathBuf::from("piper"),
            request_timeout_ms: 15_000,
        }
    }
}

/// TTS configuration used by the engine and backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsConfig {
    pub enabled: bool,
    pub engine: TtsEngineType,
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

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            engine: TtsEngineType::Silero,
            language: "ru".to_string(),
            model: "v3_1_ru".to_string(),
            speaker: "xenia".to_string(),
            sample_rate: 48_000,
            speed: 1.0,
            device: None,
            cache: TtsCacheConfig::default(),
            monitoring: TtsMonitoringConfig::default(),
            runtime: TtsRuntimeConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TtsEngineType {
    Silero,
    Piper,
    Vits,
}

/// TTS engine trait.
#[async_trait]
pub trait TtsBackend: Send + Sync {
    async fn synthesize(&self, text: &str) -> KlarnetResult<Vec<u8>>;
    fn name(&self) -> &str;
}

/// Main TTS engine orchestrating a backend, playback and caching.
pub struct TtsEngine {
    config: TtsConfig,
    backend: Box<dyn TtsBackend>,
    player: AudioPlayer,
    cache: Arc<RwLock<TtsCache>>,
}

struct TtsCache {
    entries: HashMap<String, Vec<u8>>,
    order: VecDeque<String>,
    max_size: usize,
}

impl TtsCache {
    fn new(max_size: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            max_size,
        }
    }

    fn get(&mut self, key: &str) -> Option<Vec<u8>> {
        if let Some(data) = self.entries.get(key) {
            let data = data.clone();
            self.touch(key);
            Some(data)
        } else {
            None
        }
    }

    fn insert(&mut self, key: String, data: Vec<u8>) {
        if self.max_size == 0 {
            return;
        }

        if self.entries.contains_key(&key) {
            self.entries.insert(key.clone(), data);
            self.touch(&key);
            return;
        }

        self.entries.insert(key.clone(), data);
        self.order.push_back(key.clone());

        if self.entries.len() > self.max_size {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
    }

    fn touch(&mut self, key: &str) {
        if let Some(position) = self.order.iter().position(|k| k == key) {
            self.order.remove(position);
            self.order.push_back(key.to_string());
        }
    }
}

impl TtsEngine {
    pub async fn new(config: TtsConfig) -> KlarnetResult<Self> {
        if !config.enabled {
            return Err(KlarnetError::Action(
                "TTS is disabled in configuration".to_string(),
            ));
        }
        let backend: Box<dyn TtsBackend> = match config.engine {
            TtsEngineType::Silero => Box::new(silero::SileroTts::new(config.clone()).await?),
            TtsEngineType::Piper => Box::new(piper::PiperTts::new(config.clone()).await?),
            TtsEngineType::Vits => {
                return Err(KlarnetError::Action(
                    "VITS backend is not implemented".to_string(),
                ));
            }
        };

        if config.cache.enabled {
            if let Err(err) = std::fs::create_dir_all(&config.cache.directory) {
                return Err(KlarnetError::Io(err));
            }
        }

        let player = AudioPlayer::new(config.device.as_deref())?;

        let cache = Arc::new(RwLock::new(TtsCache::new(config.cache.max_entries)));

        Ok(Self {
            config,
            backend,
            player,
            cache,
        })
    }

    pub async fn speak(&self, text: &str) -> KlarnetResult<()> {
        if text.trim().is_empty() {
            return Ok(());
        }

        info!(engine = %self.backend.name(), "TTS synthesis requested");

        let cache_key = format!("{}:{}", self.backend.name(), text);
        let mut from_cache = false;

        let cached_audio = if self.config.cache.enabled {
            let mut cache = self.cache.write();
            let hit = cache.get(&cache_key);
            if hit.is_some() {
                from_cache = true;
            }
            hit
        } else {
            None
        };

        let (audio_data, synthesis_latency) = if let Some(data) = cached_audio {
            debug!("Using cached TTS audio");
            (data, Duration::from_millis(0))
        } else {
            let started_at = Instant::now();
            let data = self.backend.synthesize(text).await?;
            let latency = started_at.elapsed();

            if self.config.cache.enabled {
                let mut cache = self.cache.write();
                cache.insert(cache_key.clone(), data.clone());
            }

            (data, latency)
        };

        if audio_data.is_empty() {
            return Err(KlarnetError::Audio(
                "Synthesis returned empty PCM buffer".to_string(),
            ));
        }

        let quality = AudioQualityReport::from_pcm(&audio_data, self.config.sample_rate);

        if self.config.monitoring.enabled && !from_cache {
            if synthesis_latency.as_millis() as u64 > self.config.monitoring.max_latency_ms {
                warn!(
                    latency_ms = synthesis_latency.as_millis() as u64,
                    max_latency_ms = self.config.monitoring.max_latency_ms,
                    "TTS synthesis latency above configured threshold"
                );
            }

            if quality.rms < self.config.monitoring.min_rms {
                warn!(
                    rms = quality.rms,
                    min_rms = self.config.monitoring.min_rms,
                    "Synthesized audio RMS below threshold"
                );
            }
        }

        debug!(
            duration_ms = quality.duration_ms,
            rms = quality.rms,
            peak = quality.peak,
            cached = from_cache,
            "TTS synthesis metrics"
        );

        self.player
            .play(&audio_data, self.config.sample_rate)
            .await?;

        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct AudioQualityReport {
    rms: f32,
    peak: f32,
    duration_ms: u64,
}

impl AudioQualityReport {
    fn from_pcm(pcm: &[u8], sample_rate: u32) -> Self {
        if pcm.is_empty() || sample_rate == 0 {
            return Self {
                rms: 0.0,
                peak: 0.0,
                duration_ms: 0,
            };
        }

        let mut sum_squares = 0.0f64;
        let mut peak = 0.0f32;
        let mut samples = 0u64;

        for chunk in pcm.chunks_exact(2) {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            let normalized = sample as f32 / i16::MAX as f32;
            peak = peak.max(normalized.abs());
            sum_squares += (normalized as f64).powi(2);
            samples += 1;
        }

        let rms = if samples == 0 {
            0.0
        } else {
            (sum_squares / samples as f64).sqrt() as f32
        };

        let duration_ms = if samples == 0 {
            0
        } else {
            (samples * 1_000 / sample_rate as u64) as u64
        };

        Self {
            rms,
            peak,
            duration_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AudioQualityReport;

    #[test]
    fn quality_report_handles_empty_pcm() {
        let report = AudioQualityReport::from_pcm(&[], 16_000);
        assert_eq!(report.rms, 0.0);
        assert_eq!(report.peak, 0.0);
        assert_eq!(report.duration_ms, 0);
    }

    #[test]
    fn quality_report_computes_metrics() {
        let samples = vec![i16::MAX / 2; 48_000];
        let pcm = samples
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect::<Vec<_>>();
        let report = AudioQualityReport::from_pcm(&pcm, 48_000);

        assert!(report.rms > 0.0);
        assert!(report.peak > 0.4);
        assert_eq!(report.duration_ms, 1_000);
    }
}

impl TtsEngine {
    #[doc(hidden)]
    #[allow(dead_code)]
    pub fn cached_pcm_for_test(&self, text: &str) -> Option<Vec<u8>> {
        let cache_key = format!("{}:{}", self.backend.name(), text);
        self.cache.read().entries.get(&cache_key).cloned()
    }
}