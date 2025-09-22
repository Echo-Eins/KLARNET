// crates/tts/src/lib.rs

use async_trait::async_trait;
use klarnet_core::{KlarnetError, KlarnetResult};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::process::Command;
use tracing::{debug, error, info};

pub mod silero;
pub mod piper;
pub mod player;

use player::AudioPlayer;

/// TTS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsConfig {
    pub enabled: bool,
    pub engine: TtsEngineType,
    pub model: String,
    pub speaker: String,
    pub sample_rate: u32,
    pub speed: f32,
    pub device: String,
    pub cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TtsEngineType {
    Silero,
    Piper,
    Vits,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            engine: TtsEngineType::Silero,
            model: "v3_1_ru".to_string(),
            speaker: "xenia".to_string(),
            sample_rate: 48000,
            speed: 1.0,
            device: "cpu".to_string(),
            cache_dir: Some(PathBuf::from("cache/tts")),
        }
    }
}

/// TTS engine trait
#[async_trait]
pub trait TtsBackend: Send + Sync {
    async fn synthesize(&self, text: &str) -> KlarnetResult<Vec<u8>>;
    fn name(&self) -> &str;
}

/// Main TTS engine
pub struct TtsEngine {
    config: TtsConfig,
    backend: Box<dyn TtsBackend>,
    player: AudioPlayer,
    cache: Arc<RwLock<TtsCache>>,
}

struct TtsCache {
    entries: std::collections::HashMap<String, Vec<u8>>,
    max_size: usize,
}

impl TtsEngine {
    pub async fn new(config: TtsConfig) -> KlarnetResult<Self> {
        let backend: Box<dyn TtsBackend> = match config.engine {
            TtsEngineType::Silero => {
                Box::new(silero::SileroTts::new(config.clone()).await?)
            }
            TtsEngineType::Piper => {
                Box::new(piper::PiperTts::new(config.clone()).await?)
            }
            _ => {
                return Err(KlarnetError::Action("TTS engine not implemented".to_string()));
            }
        };

        let player = AudioPlayer::new()?;

        let cache = Arc::new(RwLock::new(TtsCache {
            entries: std::collections::HashMap::new(),
            max_size: 100,
        }));

        // Create cache directory if specified
        if let Some(cache_dir) = &config.cache_dir {
            std::fs::create_dir_all(cache_dir)
                .map_err(|e| KlarnetError::Io(e))?;
        }

        Ok(Self {
            config,
            backend,
            player,
            cache,
        })
    }

    pub async fn speak(&self, text: &str) -> KlarnetResult<()> {
        if text.is_empty() {
            return Ok(());
        }

        info!("TTS: {}", text);

        // Check cache
        let cache_key = format!("{}:{}", self.backend.name(), text);
        let audio_data = {
            let cache = self.cache.read();
            cache.entries.get(&cache_key).cloned()
        };

        let audio_data = if let Some(data) = audio_data {
            debug!("Using cached TTS audio");
            data
        } else {
            // Synthesize new audio
            let data = self.backend.synthesize(text).await?;

            // Add to cache
            {
                let mut cache = self.cache.write();
                if cache.entries.len() >= cache.max_size {
                    // Remove oldest entry
                    if let Some(key) = cache.entries.keys().next().cloned() {
                        cache.entries.remove(&key);
                    }
                }
                cache.entries.insert(cache_key, data.clone());
            }

            data
        };

        // Play audio
        self.player.play(&audio_data).await?;

        Ok(())
    }
}