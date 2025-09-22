use async_trait::async_trait;
use std::process::Command as ProcessCommand;

use crate::{ActionHandler, ActionResult, SecurityConfig};
use klarnet_core::{KlarnetError, KlarnetResult, LocalCommand};

pub struct SystemActions {
    security: SecurityConfig,
}

impl SystemActions {
    pub fn new(security: SecurityConfig) -> KlarnetResult<Self> {
        Ok(Self { security })
    }

    fn execute_system_command(&self, cmd: &str, args: &[String]) -> KlarnetResult<String> {
        if !self.security.allow_system_commands {
            return Err(KlarnetError::Action(
                "System commands are disabled".to_string(),
            ));
        }

        let mut command = ProcessCommand::new(cmd);
        for arg in args {
            command.arg(arg);
        }

        let output = command
            .output()
            .map_err(|e| KlarnetError::Action(format!("Failed to execute command: {}", e)))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(KlarnetError::Action(
                String::from_utf8_lossy(&output.stderr).to_string(),
            ))
        }
    }
}

#[async_trait]
impl ActionHandler for SystemActions {
    async fn can_handle(&self, action: &str) -> bool {
        action.starts_with("system.")
    }

    async fn execute(&self, command: &LocalCommand) -> KlarnetResult<ActionResult> {
        match command.action.as_str() {
            "system.open_app" => {
                let app_name = command
                    .parameters
                    .get("app_name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| KlarnetError::Action("App name not provided".to_string()))?;

                let (program, args): (&str, Vec<String>) = if cfg!(target_os = "windows") {
                    (
                        "cmd",
                        vec!["/c".to_string(), "start".to_string(), app_name.to_string()],
                    )
                } else if cfg!(target_os = "macos") {
                    ("open", vec!["-a".to_string(), app_name.to_string()])
                } else {
                    (app_name, Vec::new())
                };

                self.execute_system_command(program, &args)?;
                Ok(ActionResult::success_with_message(format!(
                    "Приложение '{}' запущено",
                    app_name
                )))
            }

            "system.volume" => {
                let level = command
                    .parameters
                    .get("level")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| KlarnetError::Action("Volume level not provided".to_string()))?;

                let (program, args): (&str, Vec<String>) = if cfg!(target_os = "windows") {
                    (
                        "nircmd",
                        vec!["setsysvolume".to_string(), (level * 655).to_string()],
                    )
                } else if cfg!(target_os = "macos") {
                    (
                        "osascript",
                        vec![
                            "-e".to_string(),
                            format!("set volume output volume {}", level),
                        ],
                    )
                } else {
                    (
                        "amixer",
                        vec![
                            "set".to_string(),
                            "Master".to_string(),
                            format!("{}%", level),
                        ],
                    )
                };

                self.execute_system_command(program, &args)?;
                Ok(ActionResult::success_with_message(format!(
                    "Громкость установлена на {}%",
                    level
                )))
            }

            "system.lock" => {
                let (program, args): (&str, Vec<String>) = if cfg!(target_os = "windows") {
                    ("rundll32", vec!["user32.dll,LockWorkStation".to_string()])
                } else if cfg!(target_os = "macos") {
                    ("pmset", vec!["displaysleepnow".to_string()])
                } else {
                    ("loginctl", vec!["lock-session".to_string()])
                };

                self.execute_system_command(program, &args)?;
                Ok(ActionResult::success())
            }

            _ => Err(KlarnetError::Action(format!(
                "Unknown system action: {}",
                command.action
            ))),
        }
    }

    fn name(&self) -> &str {
        "SystemActions"
    }
}
