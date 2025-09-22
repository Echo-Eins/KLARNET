// crates/actions/src/lib.rs

use async_trait::async_trait;
use klarnet_core::{KlarnetResult, LocalCommand};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{error, info, warn};

pub mod custom;
pub mod smart_home;
pub mod system;
pub mod web;

use custom::CustomActions;
use smart_home::SmartHomeActions;
use system::SystemActions;
use web::WebActions;

/// Action configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionsConfig {
    pub enabled_modules: Vec<String>,
    pub scripts_dir: Option<String>,
    pub smart_home: Option<SmartHomeConfig>,
    pub security: SecurityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartHomeConfig {
    pub api_url: String,
    pub api_token_env: String,
    pub timeout_s: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub allow_system_commands: bool,
    pub require_confirmation: Vec<String>,
    pub blocked_commands: Vec<String>,
}

impl Default for ActionsConfig {
    fn default() -> Self {
        Self {
            enabled_modules: vec!["system".to_string(), "smart_home".to_string()],
            scripts_dir: Some("scripts/actions".to_string()),
            smart_home: None,
            security: SecurityConfig {
                allow_system_commands: true,
                require_confirmation: vec!["shutdown".to_string(), "restart".to_string()],
                blocked_commands: vec![],
            },
        }
    }
}

/// Action executor trait
#[async_trait]
pub trait ActionHandler: Send + Sync {
    async fn can_handle(&self, action: &str) -> bool;
    async fn execute(&self, command: &LocalCommand) -> KlarnetResult<ActionResult>;
    fn name(&self) -> &str;
}

/// Action execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionResult {
    pub success: bool,
    pub message: Option<String>,
    pub data: Option<Value>,
    pub speak_response: bool,
}

impl ActionResult {
    pub fn success() -> Self {
        Self {
            success: true,
            message: None,
            data: None,
            speak_response: false,
        }
    }

    pub fn success_with_message(message: String) -> Self {
        Self {
            success: true,
            message: Some(message),
            data: None,
            speak_response: true,
        }
    }

    pub fn failure(message: String) -> Self {
        Self {
            success: false,
            message: Some(message),
            data: None,
            speak_response: true,
        }
    }
}

/// Main action executor
pub struct ActionExecutor {
    config: ActionsConfig,
    handlers: Vec<Box<dyn ActionHandler>>,
    metrics: Arc<RwLock<ActionMetrics>>,
    confirmation_pending: Arc<RwLock<HashMap<String, std::time::Instant>>>,
}

#[derive(Debug, Default)]
struct ActionMetrics {
    total_executed: u64,
    successful: u64,
    failed: u64,
    blocked: u64,
    average_execution_time_ms: f64,
}

impl ActionExecutor {
    pub async fn new() -> KlarnetResult<Self> {
        Self::with_config(ActionsConfig::default()).await
    }

    pub async fn with_config(config: ActionsConfig) -> KlarnetResult<Self> {
        let mut handlers: Vec<Box<dyn ActionHandler>> = Vec::new();

        // Initialize enabled modules
        for module in &config.enabled_modules {
            match module.as_str() {
                "system" => {
                    handlers.push(Box::new(SystemActions::new(config.security.clone())?));
                }
                "smart_home" => {
                    if let Some(sh_config) = &config.smart_home {
                        handlers.push(Box::new(SmartHomeActions::new(sh_config.clone()).await?));
                    }
                }
                "web" => {
                    handlers.push(Box::new(WebActions::new()));
                }
                "custom" => {
                    if let Some(scripts_dir) = &config.scripts_dir {
                        handlers.push(Box::new(CustomActions::new(scripts_dir.clone())?));
                    }
                }
                _ => warn!("Unknown action module: {}", module),
            }
        }

        Ok(Self {
            config,
            handlers,
            metrics: Arc::new(RwLock::new(ActionMetrics::default())),
            confirmation_pending: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn execute(&self, command: LocalCommand) -> KlarnetResult<ActionResult> {
        let start = std::time::Instant::now();

        // Check if action is blocked
        if self
            .config
            .security
            .blocked_commands
            .contains(&command.action)
        {
            self.metrics.write().blocked += 1;
            return Ok(ActionResult::failure(format!(
                "Действие '{}' заблокировано",
                command.action
            )));
        }

        // Check if confirmation is required
        if self
            .config
            .security
            .require_confirmation
            .contains(&command.action)
        {
            if !self.check_confirmation(&command) {
                return Ok(ActionResult::success_with_message(format!(
                    "Подтвердите выполнение команды '{}'",
                    command.action
                )));
            }
        }

        // Find handler for the action
        for handler in &self.handlers {
            if handler.can_handle(&command.action).await {
                info!(
                    "Executing action '{}' with handler '{}'",
                    command.action,
                    handler.name()
                );

                let result = match handler.execute(&command).await {
                    Ok(res) => {
                        self.metrics.write().successful += 1;
                        res
                    }
                    Err(e) => {
                        error!("Action execution failed: {}", e);
                        self.metrics.write().failed += 1;
                        ActionResult::failure(format!("Ошибка выполнения: {}", e))
                    }
                };

                // Update metrics
                let execution_time = start.elapsed().as_millis() as f64;
                let mut metrics = self.metrics.write();
                metrics.total_executed += 1;
                metrics.average_execution_time_ms = (metrics.average_execution_time_ms
                    * (metrics.total_executed - 1) as f64
                    + execution_time)
                    / metrics.total_executed as f64;

                return Ok(result);
            }
        }

        warn!("No handler found for action: {}", command.action);
        Ok(ActionResult::failure(format!(
            "Не найден обработчик для команды '{}'",
            command.action
        )))
    }

    fn check_confirmation(&self, command: &LocalCommand) -> bool {
        let mut pending = self.confirmation_pending.write();

        // Check if we have a pending confirmation
        if let Some(timestamp) = pending.get(&command.action) {
            if timestamp.elapsed() < std::time::Duration::from_secs(30) {
                // Confirmation received within timeout
                pending.remove(&command.action);
                return true;
            }
        }

        // Add to pending confirmations
        pending.insert(command.action.clone(), std::time::Instant::now());

        false
    }
}
