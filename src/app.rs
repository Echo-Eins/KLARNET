// src/app.rs
use std::time::Duration;

use anyhow::{anyhow, Result};
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
    #[serde(default = "default_assistant_name")]
    pub assistant_name: String,
    /// Low level audio configuration shared with the pipeline.
    #[serde(default)]
    pub audio: AudioConfig,
    /// Configuration for the runtime pipeline.
    #[serde(default)]
    pub pipeline: PipelineConfig,
    /// Maximum time allowed for graceful shutdown.
    #[serde(default = "default_shutdown_timeout_ms")]
    pub shutdown_timeout_ms: u64,
}
fn default_assistant_name() -> String {
    "KLARNET".to_string()
}

fn default_shutdown_timeout_ms() -> u64 {
    5_000
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            assistant_name: default_assistant_name(),
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

        self.pipeline.start().await.map_err(|err| anyhow!(err))?;

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
            Ok(result) => result.map_err(|err| anyhow!(err)),Ok(result) => result.map_err(|err| anyhow!(err)),
            Err(_) => {
                error!("Pipeline stop timed out after {:?}", shutdown_timeout);
                Err(anyhow!("graceful shutdown timed out"))
            }
        }
    }
}