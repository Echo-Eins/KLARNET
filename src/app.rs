use std::sync::Arc;
use std::collections::VecDeque;
use std::time::Duration;

use llm_connector::{
    CompletionRequest, LlmConfig, LlmConnector, Message as LlmMessage, Role as LlmRole,
};

use actions::{ActionExecutor, ActionsConfig};
use anyhow::{anyhow, Result};
use klarnet_core::{AudioConfig, CommandType};
use serde::{Deserialize, Serialize};
use tokio::signal;
use tokio::task::JoinHandle;
use tokio::time::{sleep, timeout};
use tracing::{error, info, warn};
use tts::{TtsConfig, TtsEngine};

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
    #[serde(default)]
    pub llm: LlmAppConfig,
    #[serde(default = "default_tts_engine_config")]
    pub tts: TtsConfig,
    #[serde(default = "default_tts_retry_attempts")]
    pub tts_retry_attempts: u32,
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
            llm: LlmAppConfig::default(),
            tts: default_tts_engine_config(),
            tts_retry_attempts: default_tts_retry_attempts(),
            shutdown_timeout_ms: default_shutdown_timeout_ms(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmAppConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub connector: LlmConfig,
    #[serde(default = "default_llm_system_prompt")]
    pub system_prompt: String,
    #[serde(default = "default_llm_history_size")]
    pub max_history_messages: usize,
}

impl Default for LlmAppConfig {
    fn default() -> Self {
        let mut connector = LlmConfig::default();
        connector.model = "x-ai/grok-4-fast:free".to_string();
        Self {
            enabled: false,
            connector,
            system_prompt: default_llm_system_prompt(),
            max_history_messages: default_llm_history_size(),
        }
    }
}

fn default_tts_engine_config() -> TtsConfig {
    let mut config = TtsConfig::default();
    config.enabled = false;
    config
}

fn default_tts_retry_attempts() -> u32 {
    3
}

fn default_llm_system_prompt() -> String {
    "Ты — голосовой ассистент Кларнет. Отвечай на вопросы пользователя по-русски, оставайся дружелюбным и точным. Используй контекст последних реплик для лучшего понимания.".to_string()
}

fn default_llm_history_size() -> usize {
    10
}

pub struct KlarnetApp {
    config: AppConfig,
    pipeline: AudioPipeline,
    action_executor: Arc<ActionExecutor>,
    event_tasks: Vec<JoinHandle<()>>,
    llm_connector: Option<Arc<LlmConnector>>,
    tts_engine: Option<Arc<TtsEngine>>,
    conversation_history: ConversationHistory,
}

impl KlarnetApp {
    pub async fn new(config: AppConfig) -> Result<Self> {
        let pipeline = AudioPipeline::new(config.pipeline.clone(), config.audio.clone());
        let action_executor = Arc::new(
            ActionExecutor::with_config(config.actions.clone())
                .await
                .map_err(|err| anyhow!(err))?,
        );

        let llm_connector = if config.llm.enabled {
            info!(
                "Initialising LLM connector with model '{}'.",
                config.llm.connector.model
            );
            Some(Arc::new(
                LlmConnector::new(config.llm.connector.clone())
                    .await
                    .map_err(|err| anyhow!(err))?,
            ))
        } else {
            info!("LLM connector is disabled in configuration");
            None
        };

        let conversation_history = ConversationHistory::new(config.llm.max_history_messages);

        let tts_engine = if config.tts.enabled {
            info!(engine = ?config.tts.engine, "Initialising TTS engine");
            match TtsEngine::new(config.tts.clone()).await {
                Ok(engine) => {
                    info!("TTS engine initialised successfully");
                    Some(Arc::new(engine))
                }
                Err(err) => {
                    error!("Failed to initialise TTS engine: {err}");
                    None
                }
            }
        } else {
            info!("TTS engine is disabled in configuration");
            None
        };

        Ok(Self {
            config,
            pipeline,
            action_executor,
            event_tasks: Vec::new(),
            llm_connector,
            tts_engine,
            conversation_history,
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
            let history = self.conversation_history.clone();
            let handle = tokio::spawn(async move {
                while let Some(transcript) = stt_rx.recv().await {
                    if transcript.full_text.is_empty() {
                        info!(assistant = %assistant, "Received empty transcript");
                    } else {
                        info!(assistant = %assistant, text = %transcript.full_text, "Speech recognised");
                        history.record_user(&transcript.full_text).await;
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
            let llm_connector = self.llm_connector.clone();
            let llm_settings = self.config.llm.clone();
            let tts_engine = self.tts_engine.clone();
            let tts_retry_attempts = self.config.tts_retry_attempts;
            let history = self.conversation_history.clone();
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
                            let tts_clone = tts_engine.clone();
                            match executor.execute(command).await {
                                Ok(outcome) => {
                                    if outcome.success {
                                        if let Some(message) = outcome.message.as_ref() {
                                            info!(assistant = %assistant, action = %action_name, response = %message, "Command executed successfully");
                                        } else {
                                            info!(assistant = %assistant, action = %action_name, "Command executed successfully");
                                        }

                                        if outcome.speak_response {
                                            let speech =
                                                outcome.message.clone().unwrap_or_else(|| {
                                                    format!(
                                                        "Команда '{}' выполнена успешно.",
                                                        action_name.as_str()
                                                    )
                                                });

                                            if let Some(tts) = tts_clone.clone() {
                                                speak_with_retry(tts, speech, tts_retry_attempts)
                                                    .await;
                                            }
                                        }

                                    } else {
                                        warn!(assistant = %assistant, action = %action_name, message = ?outcome.message, "Command reported failure");

                                        if outcome.speak_response {
                                            let speech =
                                                outcome.message.clone().unwrap_or_else(|| {
                                                    format!(
                                                        "Команду '{}' выполнить не удалось.",
                                                        action_name.as_str()
                                                    )
                                                });

                                            if let Some(tts) = tts_clone.clone() {
                                                speak_with_retry(tts, speech, tts_retry_attempts)
                                                    .await;
                                            }
                                        }
                                    }
                                }
                                Err(err) => {
                                    error!(assistant = %assistant, action = %action_name, "Command execution failed: {err}");
                                    if let Some(tts) = tts_clone {
                                        speak_with_retry(
                                            tts,
                                            "Не удалось выполнить команду, произошла ошибка."
                                                .to_string(),
                                            tts_retry_attempts,
                                        )
                                            .await;
                                    }
                                }
                            }
                        }
                        CommandType::LlmRequired(description) => {
                            info!(assistant = %assistant, requirement = %description, "Command requires LLM processing");

                            let user_input = if result.transcript.trim().is_empty() {
                                description.clone()
                            } else {
                                result.transcript.clone()
                            };

                            history.record_user(&user_input).await;

                            if let Some(connector) = llm_connector.clone() {
                                let mut conversation = history.snapshot().await;
                                let mut messages = Vec::with_capacity(conversation.len() + 1);
                                messages.push(LlmMessage {
                                    role: LlmRole::System,
                                    content: llm_settings.system_prompt.clone(),
                                });
                                messages.extend(conversation.drain(..));

                                let request = CompletionRequest {
                                    messages,
                                    max_tokens: Some(llm_settings.connector.max_tokens),
                                    temperature: Some(llm_settings.connector.temperature),
                                    top_p: Some(llm_settings.connector.top_p),
                                    stop: None,
                                    functions: None,
                                };

                                match connector.complete(request).await {
                                    Ok(response) => {
                                        let answer = response.content.trim().to_string();
                                        if answer.is_empty() {
                                            warn!(assistant = %assistant, "LLM response was empty");
                                        } else {
                                            info!(assistant = %assistant, "LLM response ready");
                                            history.record_assistant(&answer).await;
                                            if let Some(tts) = tts_engine.clone() {
                                                speak_with_retry(
                                                    tts,
                                                    answer.clone(),
                                                    tts_retry_attempts,
                                                )
                                                    .await;
                                            }
                                        }
                                    }
                                    Err(err) => {
                                        error!(assistant = %assistant, "LLM request failed: {err}");
                                        let fallback = "Извините, не удалось получить ответ от языковой модели.".to_string();
                                        history.record_assistant(&fallback).await;
                                        if let Some(tts) = tts_engine.clone() {
                                            speak_with_retry(
                                                tts,
                                                fallback.clone(),
                                                tts_retry_attempts,
                                            )
                                                .await;
                                        }
                                    }
                                }
                            } else {
                                warn!(assistant = %assistant, "LLM connector is not configured");
                                let fallback =
                                    "Извините, модуль генерации ответов недоступен.".to_string();
                                history.record_assistant(&fallback).await;
                                if let Some(tts) = tts_engine.clone() {
                                    speak_with_retry(tts, fallback.clone(), tts_retry_attempts)
                                        .await;
                                }
                            }
                        }
                        CommandType::Unknown => {
                            warn!(assistant = %assistant, "Unable to determine an actionable intent");

                            let fallback = "Извините, я не поняла команду.".to_string();
                            history.record_assistant(&fallback).await;
                            if let Some(tts) = tts_engine.clone() {
                                speak_with_retry(tts, fallback.clone(), tts_retry_attempts).await;
                            }
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

#[derive(Clone)]
struct ConversationHistory {
    messages: Arc<tokio::sync::Mutex<VecDeque<LlmMessage>>>,
    capacity: usize,
}

impl ConversationHistory {
    fn new(capacity: usize) -> Self {
        Self {
            messages: Arc::new(tokio::sync::Mutex::new(VecDeque::new())),
            capacity,
        }
    }

    async fn record_user(&self, text: &str) {
        self.push_message(LlmRole::User, text).await;
    }

    async fn record_assistant(&self, text: &str) {
        self.push_message(LlmRole::Assistant, text).await;
    }

    async fn push_message(&self, role: LlmRole, text: &str) {
        let content = text.trim();
        if content.is_empty() {
            return;
        }

        let mut guard = self.messages.lock().await;

        if let Some(last) = guard.back() {
            if roles_equal(&last.role, &role) && last.content == content {
                return;
            }
        }

        if self.capacity > 0 {
            while guard.len() >= self.capacity {
                guard.pop_front();
            }
        }

        guard.push_back(LlmMessage {
            role,
            content: content.to_string(),
        });
    }

    async fn snapshot(&self) -> Vec<LlmMessage> {
        let guard = self.messages.lock().await;
        guard.iter().cloned().collect()
    }
}

fn roles_equal(left: &LlmRole, right: &LlmRole) -> bool {
    matches!(
        (left, right),
        (LlmRole::System, LlmRole::System)
            | (LlmRole::User, LlmRole::User)
            | (LlmRole::Assistant, LlmRole::Assistant)
            | (LlmRole::Function, LlmRole::Function)
    )
}

async fn speak_with_retry(engine: Arc<TtsEngine>, text: String, attempts: u32) {
    let retries = attempts.max(1);
    for attempt in 1..=retries {
        match engine.speak(&text).await {
            Ok(_) => return,
            Err(err) => {
                warn!(attempt, retries, "TTS playback failed: {err}");
                if attempt < retries {
                    let delay = Duration::from_millis(250 * attempt as u64);
                    sleep(delay).await;
                }
            }
        }
    }

    error!("TTS playback failed after {retries} attempts");
}