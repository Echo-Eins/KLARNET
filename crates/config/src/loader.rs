use std::fs;
use std::path::Path;

use serde_json::Value;
use tracing::info;

use crate::{ConfigValidator, KlarnetConfig};
use klarnet_core::{KlarnetError, KlarnetResult};

pub struct ConfigLoader;

impl ConfigLoader {
    /// Load configuration from a TOML file on disk.
    pub fn load_from_file(path: &Path) -> KlarnetResult<KlarnetConfig> {
        let content = fs::read_to_string(path)
            .map_err(|e| KlarnetError::Config(format!("Failed to read config {path:?}: {e}")))?;

        let config: KlarnetConfig = toml::from_str(&content)
            .map_err(|e| KlarnetError::Config(format!("Failed to parse config {path:?}: {e}")))?;

        Ok(config)
    }

    /// Persist configuration to disk in a human-readable TOML form.
    pub fn save_to_file(path: &Path, config: &KlarnetConfig) -> KlarnetResult<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                KlarnetError::Config(format!("Failed to create config dir {parent:?}: {e}"))
            })?;
        }

        let content = toml::to_string_pretty(config)
            .map_err(|e| KlarnetError::Config(format!("Failed to serialise config: {e}")))?;

        fs::write(path, content)
            .map_err(|e| KlarnetError::Config(format!("Failed to write config {path:?}: {e}")))?;

        Ok(())
    }

    /// Build a configuration from environment overrides.
    pub fn load_from_env() -> KlarnetResult<KlarnetConfig> {
        let mut config = KlarnetConfig::default();

        if let Ok(name) = std::env::var("KLARNET_ASSISTANT_NAME") {
            config.app.assistant_name = name;
        }
        if let Ok(language) = std::env::var("KLARNET_LANGUAGE") {
            config.app.language = language;
        }
        if let Ok(port) = std::env::var("KLARNET_API_PORT") {
            config.api.port = port
                .parse()
                .map_err(|_| KlarnetError::Config("Invalid KLARNET_API_PORT".to_string()))?;
        }

        Ok(config)
    }

    /// Merge the base configuration with an optional override file. Override values win.
    pub fn merge_configs(
        base: KlarnetConfig,
        override_path: Option<&Path>,
    ) -> KlarnetResult<KlarnetConfig> {
        let Some(path) = override_path else {
            return Ok(base);
        };

        if !path.exists() {
            info!(
                "Override config {:?} not found, using base configuration",
                path
            );
            return Ok(base);
        }

        let override_config = Self::load_from_file(path)?;

        let mut base_json = serde_json::to_value(base)
            .map_err(|e| KlarnetError::Config(format!("Failed to serialise base config: {e}")))?;
        let override_json = serde_json::to_value(override_config).map_err(|e| {
            KlarnetError::Config(format!("Failed to serialise override config: {e}"))
        })?;

        merge_json(&mut base_json, &override_json);

        let merged: KlarnetConfig = serde_json::from_value(base_json)
            .map_err(|e| KlarnetError::Config(format!("Failed to build merged config: {e}")))?;

        ConfigValidator::validate(&merged)?;
        Ok(merged)
    }
}

fn merge_json(base: &mut Value, overlay: &Value) {
    match (base, overlay) {
        (Value::Object(base_map), Value::Object(overlay_map)) => {
            for (key, value) in overlay_map {
                merge_json(base_map.entry(key.clone()).or_insert(Value::Null), value);
            }
        }
        (base_slot, overlay_value) => {
            *base_slot = overlay_value.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn merge_preserves_base_values() {
        let mut base = KlarnetConfig::default();
        base.api.port = 1234;
        let mut override_cfg = base.clone();
        override_cfg.api.port = 5678;
        override_cfg.app.assistant_name = "Jarvis".to_string();

        let merged = ConfigLoader::merge_configs(base.clone(), None).unwrap();
        assert_eq!(merged.api.port, base.api.port);

        let file = NamedTempFile::new().unwrap();
        ConfigLoader::save_to_file(file.path(), &override_cfg).unwrap();
        let merged = ConfigLoader::merge_configs(base, Some(file.path())).unwrap();
        assert_eq!(merged.api.port, 5678);
        assert_eq!(merged.app.assistant_name, "Jarvis");
    }

    #[test]
    fn load_and_save_roundtrip() {
        let file = NamedTempFile::new().unwrap();
        let config = KlarnetConfig::default();
        ConfigLoader::save_to_file(file.path(), &config).unwrap();
        let loaded = ConfigLoader::load_from_file(file.path()).unwrap();
        assert_eq!(config, loaded);
    }
}
