// src/main.rs

use std::{env, path::PathBuf};

use anyhow::{anyhow, Context, Result};
use klarnet_config::{ConfigLoader, ConfigValidator, KlarnetConfig};
use klarnet_core::resolve_project_path;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod app;
mod commands;
mod device;
mod pipeline;

use crate::pipeline::PipelineConfig;
use app::{AppConfig, KlarnetApp};

#[tokio::main]
async fn main() -> Result<()> {
    init_logging()?;
    info!(
        "Starting KLARNET Voice Assistant v{}",
        env!("CARGO_PKG_VERSION")
    );

    let mut config = load_config().await?;

    async fn main() -> Result<()> {
        init_logging()?;
        info!(
            "Starting KLARNET Voice Assistant v{}",
            env!("CARGO_PKG_VERSION")
        );

        let mut config = load_config().await?;

        // Интерактивный выбор аудио устройств
        device::force_interactive_device_selection(&mut config)?;

        // Тестирование выбранных устройств
        device::test_audio_devices(&config).await?;

        let mut app = KlarnetApp::new(config).await?;
        app.run().await?;

        Ok(())
    }

    let mut app = KlarnetApp::new(config).await?;
    app.run().await?;

    Ok(())
}

fn init_logging() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "klarnet=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .try_init()
        .context("failed to initialise tracing subscriber")?;
    Ok(())
}

async fn load_config() -> Result<AppConfig> {
    let override_path = config_path_override()?;
    let runtime = load_runtime_config(override_path).await?;

    let mut config = AppConfig::default();
    config.runtime = runtime.clone();
    config.pipeline = PipelineConfig::from_runtime(&runtime);
    Ok(config)
}
fn config_path_override() -> Result<Option<PathBuf>> {
    let mut args = env::args().skip(1);
    let mut override_path: Option<PathBuf> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" | "-c" => {
                let Some(value) = args.next() else {
                    return Err(anyhow!("Expected path after {}", arg));
                };
                override_path = Some(resolve_project_path(value));
            }
            _ if override_path.is_none() && !arg.starts_with('-') => {
                override_path = Some(resolve_project_path(&arg));
            }
            _ => {}
        }
    }

    Ok(override_path)
}

async fn load_runtime_config(path_override: Option<PathBuf>) -> Result<KlarnetConfig> {
    if let Some(path) =
        path_override.or_else(|| env::var("KLARNET_CONFIG").ok().map(resolve_project_path))
    {
        return read_runtime_from_path(path).await;
    }

    let default_path = resolve_project_path("config/klarnet.toml");
    if default_path.exists() {
        return read_runtime_from_path(default_path).await;
    }

    info!("No klarnet.toml found; using built-in defaults");
    let config = apply_env_overrides(KlarnetConfig::default())?;
    ConfigValidator::validate(&config)?;
    Ok(config)
}

async fn read_runtime_from_path(path: PathBuf) -> Result<KlarnetConfig> {
    let path_clone = path.clone();
    let config = tokio::task::spawn_blocking(move || ConfigLoader::load_from_file(&path_clone))
        .await
        .context("Failed to join config loading task")??;

    let config = apply_env_overrides(config)?;
    ConfigValidator::validate(&config)
        .with_context(|| format!("Invalid configuration in {:?}", path))?;

    info!("Loaded configuration from {:?}", path);
    Ok(config)
}

fn apply_env_overrides(mut config: KlarnetConfig) -> Result<KlarnetConfig> {
    if let Ok(name) = env::var("KLARNET_ASSISTANT_NAME") {
        if !name.trim().is_empty() {
            config.app.assistant_name = name;
        }
    }
    if let Ok(language) = env::var("KLARNET_LANGUAGE") {
        if !language.trim().is_empty() {
            config.app.language = language;
        }
    }
    if let Ok(mode) = env::var("KLARNET_MODE") {
        if !mode.trim().is_empty() {
            config.app.mode = mode;
        }
    }
    if let Ok(port) = env::var("KLARNET_API_PORT") {
        let port = port
            .trim()
            .parse()
            .map_err(|_| anyhow!("KLARNET_API_PORT must be a valid port number"))?;
        config.api.port = port;
    }

    Ok(config)
}
