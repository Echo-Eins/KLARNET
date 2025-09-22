// src/main.rs

use std::path::Path;

use anyhow::{Context, Result};
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
    let config_path = Path::new("config/app.toml");
    if !config_path.exists() {
        info!(
            "Configuration file {:?} not found. Using defaults.",
            config_path
        );
        return Ok(AppConfig::default());
    }

    let contents = fs::read_to_string(config_path)
        .await
        .with_context(|| format!("Failed to read configuration from {:?}", config_path))?;

    let config: AppConfig = toml::from_str(&contents)
        .with_context(|| format!("Invalid configuration in {:?}", config_path))?;

    info!("Loaded configuration from {:?}", config_path);
    Ok(config)
}
