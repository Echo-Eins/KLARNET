// crates/config/src/validator.rs

pub struct ConfigValidator;

impl ConfigValidator {
    pub fn validate(config: &KlarnetConfig) -> KlarnetResult<()> {
        // Validate audio settings
        if config.audio.sample_rate == 0 {
            return Err(KlarnetError::Config("Invalid sample rate".to_string()));
        }
        if config.audio.channels == 0 || config.audio.channels > 2 {
            return Err(KlarnetError::Config("Invalid channel count".to_string()));
        }

        // Validate VAD settings
        if config.vad.aggressiveness > 3 {
            return Err(KlarnetError::Config("VAD aggressiveness must be 0-3".to_string()));
        }
        if ![10, 20, 30].contains(&config.vad.frame_duration_ms) {
            return Err(KlarnetError::Config("VAD frame duration must be 10, 20, or 30 ms".to_string()));
        }

        // Validate STT settings
        if !config.stt.model_path.exists() {
            warn!("STT model path does not exist: {:?}", config.stt.model_path);
        }
        if config.stt.beam_size == 0 {
            return Err(KlarnetError::Config("Beam size must be > 0".to_string()));
        }

        // Validate NLU settings
        if config.nlu.wake_words.is_empty() {
            warn!("No wake words configured");
        }
        if config.nlu.confidence_threshold < 0.0 || config.nlu.confidence_threshold > 1.0 {
            return Err(KlarnetError::Config("Confidence threshold must be 0.0-1.0".to_string()));
        }

        // Validate API settings
        if config.api.enabled && config.api.port == 0 {
            return Err(KlarnetError::Config("Invalid API port".to_string()));
        }

        Ok(())
    }

    pub fn validate_runtime_change(old: &KlarnetConfig, new: &KlarnetConfig) -> KlarnetResult<()> {
        // Some settings cannot be changed at runtime
        if old.audio.sample_rate != new.audio.sample_rate {
            return Err(KlarnetError::Config("Cannot change sample rate at runtime".to_string()));
        }
        if old.stt.model_path != new.stt.model_path {
            return Err(KlarnetError::Config("Cannot change model path at runtime".to_string()));
        }

        Ok(())
    }
}