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

use app::KlarnetApp;
use pipeline::AudioPipeline;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    init_logging()?;

    info!("Starting KLARNET Voice Assistant v{}", env!("CARGO_PKG_VERSION"));

    // Load configuration
    let config = load_config().await?;

    // Initialize application
    let mut app = KlarnetApp::new(config).await?;

    // Start the assistant
    app.run().await?;

    info!("KLARNET shut down successfully");
    Ok(())
}

fn init_logging() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "klarnet=debug,info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    Ok(())
}

async fn load_config() -> Result<AppConfig> {
    // Load from config file or use defaults
    Ok(AppConfig::default())
}