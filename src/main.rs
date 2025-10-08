// src/main.rs

use std::{
    env,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Context, Result};
use tokio::fs;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod app;
mod commands;
mod pipeline;

use app::{AppConfig, KlarnetApp};

#[tokio::main]
async fn main() -> Result<()> {
    init_logging()?;
    info!(
        "Starting KLARNET Voice Assistant v{}",
        env!("CARGO_PKG_VERSION")
    );

    let config = load_config().await?;

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
    if let Some(path) = config_path_override()? {
        return load_config_from_path(path).await;
    }

    if let Ok(env_path) = env::var("KLARNET_CONFIG") {
        let trimmed = env_path.trim();
        if !trimmed.is_empty() {
            return load_config_from_path(trimmed).await;
        }
    }
    const CONFIG_CANDIDATES: [&str; 2] = ["config/app.toml", "config/klarnet.toml"];

    let config_path = CONFIG_CANDIDATES
        .iter()
        .map(Path::new)
        .find(|path| path.exists());

    let Some(config_path) = config_path else {
        info!(
            "No configuration file found at any of {:?}. Using defaults.",
            CONFIG_CANDIDATES
        );
        return Ok(AppConfig::default());
    };

    load_config_from_path(config_path).await
}

async fn load_config_from_path(path: impl AsRef<Path>) -> Result<AppConfig> {
    let path = path.as_ref().to_path_buf();

    let contents = fs::read_to_string(&path)
        .await
        .with_context(|| format!("Failed to read configuration from {:?}", path))?;

    let config: AppConfig = toml::from_str(&contents)
        .with_context(|| format!("Invalid configuration in {:?}", path))?;

    info!("Loaded configuration from {:?}", path);
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
                override_path = Some(PathBuf::from(value));
            }
            _ if override_path.is_none() && !arg.starts_with('-') => {
                override_path = Some(PathBuf::from(arg));
            }
            _ => {}
        }
    }

    Ok(override_path)
}