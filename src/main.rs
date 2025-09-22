// src/main.rs

use anyhow::Result;
use klarnet_core::{AudioConfig, AudioFrame, KlarnetResult, VadEvent};
use std::sync::Arc;
use std::time::Duration;
use tokio::signal;
use tokio::sync::mpsc;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod app;
mod pipeline;
mod commands;
use std::path::Path;

use app::KlarnetApp;
use pipeline::AudioPipeline;

#[tokio::main]
async fn main() -> Result<()> {
    init_logging()?;
    info!(
        "Starting KLARNET Voice Assistant v{}",
        env!("CARGO_PKG_VERSION")
    );

    let config = load_config().await?;

    let mut app = KlarnetApp::new(config)?;
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
        .init();
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