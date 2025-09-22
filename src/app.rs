use std::sync::Arc;
use std::time::Duration;

use actions::{ActionExecutor, ActionsConfig};
use anyhow::{anyhow, Result};
use klarnet_core::{AudioConfig, CommandType};
use serde::{Deserialize, Serialize};
use tokio::signal;
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tracing::{error, info, warn};

use crate::pipeline::{AudioPipeline, PipelineConfig};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default = "default_assistant_name")]
    pub assistant_name: String,
    #[serde(default)]
    pub audio: AudioConfig,
    #[serde(default)]
    pub pipeline: PipelineConfig,
    #[serde(default)]
    pub actions: ActionsConfig,
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
            actions: ActionsConfig::default(),
            shutdown_timeout_ms: default_shutdown_timeout_ms(),
        }
    }
}

pub struct KlarnetApp {
    config: AppConfig,
    pipeline: AudioPipeline,
    action_executor: Arc<ActionExecutor>,
    event_tasks: Vec<JoinHandle<()>>,
}

impl KlarnetApp {
    pub async fn new(config: AppConfig) -> Result<Self> {
        let pipeline = AudioPipeline::new(config.pipeline.clone(), config.audio.clone());
        let action_executor = Arc::new(
            ActionExecutor::with_config(config.actions.clone())
                .await
                .map_err(|err| anyhow!(err))?,
        );

        Ok(Self {
            config,
            pipeline,
            action_executor,
            event_tasks: Vec::new(),
        })
    }

    pub async fn run(&mut self) -> Result<()> {
        info!("Starting assistant '{}'.", self.config.assistant_name);

        self.pipeline.start().await.map_err(|err| anyhow!(err))?;
        self.spawn_event_handlers();

        self.wait_for_shutdown().await?;

        let shutdown_result = self.shutdown_pipeline().await;
        self.await_event_tasks().await;
        shutdown_result?;

        info!("Assistant '{}' stopped.", self.config.assistant_name);
        Ok(())
    }

    fn spawn_event_handlers(&mut self) {
        if let Some(mut stt_rx) = self.pipeline.take_stt_receiver() {
            let assistant = self.config.assistant_name.clone();
            let handle = tokio::spawn(async move {
                while let Some(transcript) = stt_rx.recv().await {
                    if transcript.full_text.is_empty() {
                        info!(assistant = %assistant, "Received empty transcript");
                    } else {
                        info!(assistant = %assistant, text = %transcript.full_text, "Speech recognised");
                    }
                }
            });
            self.event_tasks.push(handle);
        } else {
            warn!("STT receiver is not available; transcripts will be lost");
        }

        if let Some(mut nlu_rx) = self.pipeline.take_nlu_receiver() {
            let assistant = self.config.assistant_name.clone();
            let executor = Arc::clone(&self.action_executor);
            let handle = tokio::spawn(async move {
                while let Some(result) = nlu_rx.recv().await {
                    if result.wake_word_detected {
                        info!(assistant = %assistant, transcript = %result.transcript, "Wake word detected");
                    } else {
                        info!(assistant = %assistant, transcript = %result.transcript, "NLU processed without wake word");
                    }

                    match result.command_type {
                        CommandType::Local(command) => {
                            let action_name = command.action.clone();
                            match executor.execute(command).await {
                                Ok(outcome) => {
                                    if outcome.success {
                                        if let Some(message) = outcome.message.as_ref() {
                                            info!(assistant = %assistant, action = %action_name, response = %message, "Command executed successfully");
                                        } else {
                                            info!(assistant = %assistant, action = %action_name, "Command executed successfully");
                                        }
                                    } else {
                                        warn!(assistant = %assistant, action = %action_name, message = ?outcome.message, "Command reported failure");
                                    }
                                }
                                Err(err) => {
                                    error!(assistant = %assistant, action = %action_name, "Command execution failed: {err}");
                                }
                            }
                        }
                        CommandType::LlmRequired(description) => {
                            info!(assistant = %assistant, requirement = %description, "Command requires LLM processing");
                        }
                        CommandType::Unknown => {
                            warn!(assistant = %assistant, "Unable to determine an actionable intent");
                        }
                    }
                }
            });
            self.event_tasks.push(handle);
        } else {
            warn!("NLU receiver is not available; intents will be lost");
        }
    }

    async fn await_event_tasks(&mut self) {
        for handle in self.event_tasks.drain(..) {
            if let Err(err) = handle.await {
                warn!("Event handler task terminated: {}", err);
            }
        }
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
            Ok(result) => result.map_err(|err| anyhow!(err)),
            Err(_) => {
                error!("Pipeline stop timed out after {:?}", shutdown_timeout);
                Err(anyhow!("graceful shutdown timed out"))
            }
        }
    }
}
