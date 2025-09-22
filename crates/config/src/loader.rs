// crates/config/src/loader.rs

pub struct ConfigLoader;

impl ConfigLoader {
    pub fn load_from_file(path: &Path) -> KlarnetResult<KlarnetConfig> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| KlarnetError::Config(format!("Failed to read config: {}", e)))?;

        let config: KlarnetConfig = toml::from_str(&content)
            .map_err(|e| KlarnetError::Config(format!("Failed to parse config: {}", e)))?;

        Ok(config)
    }

    pub fn save_to_file(path: &Path, config: &KlarnetConfig) -> KlarnetResult<()> {
        let content = toml::to_string_pretty(config)
            .map_err(|e| KlarnetError::Config(format!("Failed to serialize config: {}", e)))?;

        std::fs::write(path, content)
            .map_err(|e| KlarnetError::Config(format!("Failed to write config: {}", e)))?;

        Ok(())
    }

    pub fn load_from_env() -> KlarnetResult<KlarnetConfig> {
        let mut config = KlarnetConfig::default();

        // Override with environment variables
        if let Ok(lang) = std::env::var("KLARNET_LANGUAGE") {
            config.app.language = lang;
        }
        if let Ok(mode) = std::env::var("KLARNET_MODE") {
            config.app.mode = mode;
        }
        if let Ok(port) = std::env::var("KLARNET_API_PORT") {
            config.api.port = port.parse()
                .map_err(|_| KlarnetError::Config("Invalid API port".to_string()))?;
        }

        Ok(config)
    }

    pub fn merge_configs(base: KlarnetConfig, override_path: Option<&Path>) -> KlarnetResult<KlarnetConfig> {
        let mut config = base;

        if let Some(path) = override_path {
            let override_config = Self::load_from_file(path)?;
            // Merge logic here
            config = override_config;
        }

        Ok(config)
    }
}