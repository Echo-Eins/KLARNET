// crates/actions/src/web.rs

pub struct WebActions;

impl WebActions {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ActionHandler for WebActions {
    async fn can_handle(&self, action: &str) -> bool {
        action.starts_with("web.")
    }

    async fn execute(&self, command: &LocalCommand) -> KlarnetResult<ActionResult> {
        match command.action.as_str() {
            "web.search" => {
                let query = command.parameters.get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| KlarnetError::Action("Search query not provided".to_string()))?;

                let url = format!("https://www.google.com/search?q={}",
                                  urlencoding::encode(query));

                if cfg!(target_os = "windows") {
                    ProcessCommand::new("cmd")
                        .args(&["/c", "start", &url])
                        .spawn()
                        .map_err(|e| KlarnetError::Action(e.to_string()))?;
                } else if cfg!(target_os = "macos") {
                    ProcessCommand::new("open")
                        .arg(&url)
                        .spawn()
                        .map_err(|e| KlarnetError::Action(e.to_string()))?;
                } else {
                    ProcessCommand::new("xdg-open")
                        .arg(&url)
                        .spawn()
                        .map_err(|e| KlarnetError::Action(e.to_string()))?;
                }

                Ok(ActionResult::success_with_message(
                    format!("Поиск: {}", query)
                ))
            }

            _ => Err(KlarnetError::Action(format!("Unknown web action: {}", command.action)))
        }
    }

    fn name(&self) -> &str {
        "WebActions"
    }
}
