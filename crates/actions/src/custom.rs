// crates/actions/src/custom.rs

use std::path::PathBuf;

use crate::{ActionHandler, ActionResult};
use async_trait::async_trait;
use klarnet_core::{KlarnetError, KlarnetResult, LocalCommand};
use std::process::Command as ProcessCommand;

pub struct CustomActions {
    scripts_dir: PathBuf,
}

impl CustomActions {
    pub fn new(scripts_dir: String) -> KlarnetResult<Self> {
        let path = PathBuf::from(scripts_dir);
        if !path.exists() {
            std::fs::create_dir_all(&path).map_err(|e| {
                KlarnetError::Action(format!("Failed to create scripts dir: {}", e))
            })?;
        }

        Ok(Self { scripts_dir: path })
    }
}

#[async_trait]
impl ActionHandler for CustomActions {
    async fn can_handle(&self, action: &str) -> bool {
        action.starts_with("custom.")
    }

    async fn execute(&self, command: &LocalCommand) -> KlarnetResult<ActionResult> {
        let script_name = command
            .action
            .strip_prefix("custom.")
            .ok_or_else(|| KlarnetError::Action("Invalid custom action".to_string()))?;

        let script_path = self.scripts_dir.join(format!("{}.sh", script_name));

        if !script_path.exists() {
            return Err(KlarnetError::Action(format!(
                "Script not found: {}",
                script_name
            )));
        }

        let output = ProcessCommand::new("sh")
            .arg(&script_path)
            .output()
            .map_err(|e| KlarnetError::Action(format!("Failed to execute script: {}", e)))?;

        if output.status.success() {
            Ok(ActionResult::success_with_message(
                String::from_utf8_lossy(&output.stdout).trim().to_string(),
            ))
        } else {
            Err(KlarnetError::Action(
                String::from_utf8_lossy(&output.stderr).to_string(),
            ))
        }
    }

    fn name(&self) -> &str {
        "CustomActions"
    }
}
