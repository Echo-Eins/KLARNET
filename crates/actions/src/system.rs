// crates/actions/src/system.rs

pub struct SystemActions {
    security: SecurityConfig,
}

impl SystemActions {
    pub fn new(security: SecurityConfig) -> KlarnetResult<Self> {
        Ok(Self { security })
    }

    fn execute_system_command(&self, cmd: &str, args: &[&str]) -> KlarnetResult<String> {
        if !self.security.allow_system_commands {
            return Err(KlarnetError::Action("System commands are disabled".to_string()));
        }

        let output = ProcessCommand::new(cmd)
            .args(args)
            .output()
            .map_err(|e| KlarnetError::Action(format!("Failed to execute command: {}", e)))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(KlarnetError::Action(
                String::from_utf8_lossy(&output.stderr).to_string()
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
                let app_name = command.parameters.get("app_name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| KlarnetError::Action("App name not provided".to_string()))?;

                let cmd = if cfg!(target_os = "windows") {
                    ("cmd", vec!["/c", "start", app_name])
                } else if cfg!(target_os = "macos") {
                    ("open", vec!["-a", app_name])
                } else {
                    (app_name, vec![])
                };

                self.execute_system_command(cmd.0, &cmd.1)?;
                Ok(ActionResult::success_with_message(
                    format!("Приложение '{}' запущено", app_name)
                ))
            }

            "system.volume" => {
                let level = command.parameters.get("level")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| KlarnetError::Action("Volume level not provided".to_string()))?;

                let cmd = if cfg!(target_os = "windows") {
                    ("nircmd", vec!["setsysvolume", &(level * 655).to_string()])
                } else if cfg!(target_os = "macos") {
                    ("osascript", vec!["-e", &format!("set volume output volume {}", level)])
                } else {
                    ("amixer", vec!["set", "Master", &format!("{}%", level)])
                };

                self.execute_system_command(cmd.0, &cmd.1)?;
                Ok(ActionResult::success_with_message(
                    format!("Громкость установлена на {}%", level)
                ))
            }

            "system.lock" => {
                let cmd = if cfg!(target_os = "windows") {
                    ("rundll32", vec!["user32.dll,LockWorkStation"])
                } else if cfg!(target_os = "macos") {
                    ("pmset", vec!["displaysleepnow"])
                } else {
                    ("loginctl", vec!["lock-session"])
                };

                self.execute_system_command(cmd.0, &cmd.1)?;
                Ok(ActionResult::success())
            }

            _ => Err(KlarnetError::Action(format!("Unknown system action: {}", command.action)))
        }
    }

    fn name(&self) -> &str {
        "SystemActions"
    }
}