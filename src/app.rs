// src/app.rs
use std::time::Duration;

use anyhow::Result;
use klarnet_core::AudioConfig;
use serde::{Deserialize, Serialize};
use tokio::signal;
use tokio::time::timeout;
use tracing::{error, info};

use crate::pipeline::{AudioPipeline, PipelineConfig};

/// Application level configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Name that will be used in logs and responses.
    pub assistant_name: String,
    /// Low level audio configuration shared with the pipeline.
    pub audio: AudioConfig,
    /// Configuration for the runtime pipeline.
    pub pipeline: PipelineConfig,
    /// Maximum time allowed for graceful shutdown.
    #[serde(default = "default_shutdown_timeout_ms")]
    pub shutdown_timeout_ms: u64,
}

fn default_shutdown_timeout_ms() -> u64 {
    5_000
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            assistant_name: "KLARNET".to_string(),
            audio: AudioConfig::default(),
            pipeline: PipelineConfig::default(),
            shutdown_timeout_ms: default_shutdown_timeout_ms(),
        }
    }
}

/// Core application orchestrating the pipeline lifecycle.
pub struct KlarnetApp {
    config: AppConfig,
    pipeline: AudioPipeline,
}

impl KlarnetApp {
    pub fn new(config: AppConfig) -> Result<Self> {
        let pipeline = AudioPipeline::new(config.pipeline.clone(), config.audio.clone());

        Ok(Self { config, pipeline })
    }

    pub async fn run(&mut self) -> Result<()> {
        info!("Starting assistant '{}'.", self.config.assistant_name);

        self.pipeline.start().await?;

        self.wait_for_shutdown().await?;

        self.shutdown_pipeline().await?;
        info!("Assistant '{}' stopped.", self.config.assistant_name);
        Ok(())
    }
    async fn wait_for_shutdown(&self) -> Result<()> {
        info!("Waiting for shutdown signal (Ctrl+C)...");
        signal::ctrl_c().await?;
        info!("Shutdown signal received.");
        Ok(())
    }

    async fn shutdown_pipeline(&mut self) -> Result<()> {
        let shutdown_timeout = Duration::from_millis(self.config.shutdown_timeout_ms);
        info!("Stopping audio pipeline (timeout: {:?})", shutdown_timeout);

        let stop_future = self.pipeline.stop();
        match timeout(shutdown_timeout, stop_future).await {
            Ok(result) => result.map_err(|err| err.into()),
            Err(_) => {
                error!("Pipeline stop timed out after {:?}", shutdown_timeout);
                Err(anyhow::anyhow!("graceful shutdown timed out"))
            }
        }
    }
}