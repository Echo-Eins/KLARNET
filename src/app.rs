use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use tts::{TtsConfig, TtsEngine, TtsEngineType};

use llm_connector::{
    CompletionRequest, LlmConfig, LlmConnector, Message as LlmMessage, Role as LlmRole,
};

use actions::{ActionExecutor, ActionsConfig};
use anyhow::{anyhow, Result};
use klarnet_api::{handlers::ApiHandlers, ApiServer};
use klarnet_config::{ApiConfig as ApiRuntimeConfig, KlarnetConfig};
use klarnet_core::{AudioConfig, CommandType};
use klarnet_observability::{MetricsCollector, ObservabilityConfig};
use nlu::NluEngine;
use serde::{Deserialize, Serialize};
use tokio::signal;
use tokio::sync::{oneshot, Mutex};
use tokio::task::JoinHandle;
use tokio::time::{sleep, timeout};
use tracing::{error, info, warn};
use whisper_stt::WhisperEngine;

use crate::pipeline::{AudioPipeline, PipelineConfig};

pub struct AppConfig {
    pub runtime: KlarnetConfig,
    pub pipeline: PipelineConfig,
    pub llm: LlmAppConfig,
    pub tts_retry_attempts: u32,
    pub shutdown_timeout_ms: u64,
}

fn default_shutdown_timeout_ms() -> u64 {
    5_000
}

impl Default for AppConfig {
    fn default() -> Self {
        let runtime = KlarnetConfig::default();
        Self {
            pipeline: PipelineConfig::from_runtime(&runtime),
            llm: LlmAppConfig::default(),
            runtime,
            tts_retry_attempts: default_tts_retry_attempts(),
            shutdown_timeout_ms: default_shutdown_timeout_ms(),
        }
    }
}

impl AppConfig {
    pub fn assistant_name(&self) -> &str {
        &self.runtime.app.assistant_name
    }

    pub fn audio(&self) -> &AudioConfig {
        &self.runtime.audio
    }

    #[cfg(feature = "hardware")]
    pub fn audio_mut(&mut self) -> &mut AudioConfig {
        &mut self.runtime.audio
    }

    pub fn actions(&self) -> &ActionsConfig {
        &self.runtime.actions
    }

    pub fn tts(&self) -> &TtsConfig {
        &self.runtime.tts
    }

    #[cfg(feature = "hardware")]
    pub fn tts_mut(&mut self) -> &mut TtsConfig {
        &mut self.runtime.tts
    }

    pub fn api(&self) -> &ApiRuntimeConfig {
        &self.runtime.api
    }

    pub fn observability(&self) -> &ObservabilityConfig {
        &self.runtime.observability
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
            enabled: true,
            connector,
            system_prompt: default_llm_system_prompt(),
            max_history_messages: default_llm_history_size(),
        }
    }
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
    api_handlers: Arc<ApiHandlers>,
    api_task: Option<JoinHandle<()>>,
    api_shutdown: Option<oneshot::Sender<()>>,
}

impl KlarnetApp {
    pub async fn new(config: AppConfig) -> Result<Self> {
        let pipeline = AudioPipeline::new(config.pipeline.clone(), config.audio().clone());
        let action_executor = Arc::new(
            ActionExecutor::with_config(config.actions().clone())
                .await
                .map_err(|err| anyhow!(err))?,
        );

        let metrics = Arc::new(MetricsCollector::with_config(
            config.observability().clone(),
        ));

        let api_handlers = if config.api().enabled {
            match (
                WhisperEngine::new(config.pipeline.stt.clone()).await,
                NluEngine::new(config.pipeline.nlu.clone()).await,
            ) {
                (Ok(stt), Ok(nlu)) => Arc::new(ApiHandlers::with_engines(
                    Arc::new(Mutex::new(stt)),
                    Arc::new(nlu),
                    metrics.clone(),
                )),
                (stt_result, nlu_result) => {
                    if let Err(err) = stt_result {
                        warn!("Failed to create API STT engine: {err}");
                    }
                    if let Err(err) = nlu_result {
                        warn!("Failed to create API NLU engine: {err}");
                    }
                    Arc::new(ApiHandlers::new(metrics.clone()))
                }
            }
        } else {
            Arc::new(ApiHandlers::new(metrics.clone()))
        };

        let llm_connector = if config.llm.enabled {
            info!(
                "Initialising LLM connector with model '{}'.",
                config.llm.connector.model
            );
            match LlmConnector::new(config.llm.connector.clone()).await {
                Ok(connector) => Some(Arc::new(connector)),
                Err(err) => {
                    error!("Failed to initialise LLM connector: {err}");
                    None
                }
            }
        } else {
            info!("LLM connector is disabled in configuration");
            None
        };

        let conversation_history = ConversationHistory::new(config.llm.max_history_messages);

        let tts_engine = if config.tts().enabled {
            info!(engine = ?config.tts().engine, "Initialising TTS engine");
            match TtsEngine::new(config.tts().clone()).await {
                Ok(engine) => {
                    info!("TTS engine initialised successfully");
                    let engine = Arc::new(engine);
                    if matches!(config.tts().engine.clone(), TtsEngineType::Silero) {
                        play_silero_startup_prompt(engine.clone()).await;
                    }
                    Some(engine)
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
            api_handlers,
            api_task: None,
            api_shutdown: None,
        })
    }

    pub async fn run(&mut self) -> Result<()> {
        info!("Starting assistant '{}'.", self.config.assistant_name());

        self.start_api_server().await?;

        self.pipeline.start().await.map_err(|err| anyhow!(err))?;

        let stt_ready = self.pipeline.stt_channel_ready();
        let nlu_ready = self.pipeline.nlu_channel_ready();
        let llm_ready = self.llm_connector.is_some();

        self.spawn_event_handlers();

        if let Some(tts) = self.tts_engine.clone() {
            self.announce_startup(tts, stt_ready, nlu_ready, llm_ready)
                .await;
        } else {
            info!(
                stt_ready,
                nlu_ready, llm_ready, "Startup announcement skipped because TTS is disabled"
            );
        }

        self.wait_for_shutdown().await?;

        let shutdown_result = self.shutdown_pipeline().await;
        self.await_event_tasks().await;
        self.stop_api_server().await;
        shutdown_result?;

        info!("Assistant '{}' stopped.", self.config.assistant_name());
        Ok(())
    }

    fn spawn_event_handlers(&mut self) {
        if let Some(mut stt_rx) = self.pipeline.take_stt_receiver() {
            let assistant = self.config.assistant_name().to_string();
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
            let assistant = self.config.assistant_name().to_string();
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

    async fn announce_startup(
        &self,
        tts: Arc<TtsEngine>,
        stt_ready: bool,
        nlu_ready: bool,
        llm_ready: bool,
    ) {
        let retry_attempts = self.config.tts_retry_attempts;
        info!(
            stt_ready,
            nlu_ready, llm_ready, "Announcing assistant readiness over TTS"
        );

        let mut parts = Vec::new();
        parts.push(module_status_phrase("распознавание речи", stt_ready));
        parts.push(module_status_phrase("обработка команд", nlu_ready));
        parts.push(module_status_phrase("LLM коннектор", llm_ready));
        parts.push("синтез речи активен".to_string());

        let announcement = format!("Привет! {}. Готова к работе.", parts.join(", "));

        speak_with_retry(tts, announcement, retry_attempts).await;
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

    async fn start_api_server(&mut self) -> Result<()> {
        if !self.config.api().enabled {
            return Ok(());
        }

        let (tx, rx) = oneshot::channel();
        let shutdown = async move {
            let _ = rx.await;
        };

        let server = ApiServer::with_shared_handlers(
            self.config.api().clone(),
            self.api_handlers.clone(),
            self.config.audio().sample_rate,
        );

        if let Some(handle) = server.serve(shutdown).await? {
            self.api_task = Some(handle);
            self.api_shutdown = Some(tx);
        }

        Ok(())
    }

    async fn stop_api_server(&mut self) {
        if let Some(tx) = self.api_shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.api_task.take() {
            if let Err(err) = handle.await {
                warn!("API server task terminated: {err}");
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

async fn play_silero_startup_prompt(engine: Arc<TtsEngine>) {
    let message = "Кларнет готов к работе";
    info!("Playing Silero startup prompt");
    if let Err(err) = engine.speak(message).await {
        warn!("Failed to play Silero startup prompt: {err}");
    }
}

fn module_status_phrase(name: &str, ready: bool) -> String {
    if ready {
        format!("{} активен", name)
    } else {
        format!("{} недоступен", name)
    }
}
