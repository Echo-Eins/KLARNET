use tracing::warn;

use crate::KlarnetConfig;
use klarnet_core::{KlarnetError, KlarnetResult};

pub struct ConfigValidator;

impl ConfigValidator {
    pub fn validate(config: &KlarnetConfig) -> KlarnetResult<()> {
        if config.audio.sample_rate == 0 {
            return Err(KlarnetError::Config(
                "Audio sample rate must be greater than zero".to_string(),
            ));
        }
        if config.audio.channels == 0 || config.audio.channels > 2 {
            return Err(KlarnetError::Config(
                "Audio channels must be 1 or 2".to_string(),
            ));
        }
        if config.api.port == 0 {
            return Err(KlarnetError::Config(
                "API port must be greater than zero".to_string(),
            ));
        }
        if config.metrics.enabled && config.metrics.prometheus_port == 0 {
            return Err(KlarnetError::Config(
                "Prometheus port must be greater than zero".to_string(),
            ));
        }

        if let Err(err) = config.stt.validate() {
            warn!("Whisper configuration validation warning: {err}");
        }

        if config.nlu.wake_words.is_empty() {
            warn!("No wake words configured; assistant will not react to keywords");
        }

        Ok(())
    }
}
