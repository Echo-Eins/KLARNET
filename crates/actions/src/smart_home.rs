// crates/actions/src/smart_home.rs

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};

use crate::{ActionHandler, ActionResult, SmartHomeConfig};
use klarnet_core::{KlarnetError, KlarnetResult, LocalCommand};

pub struct SmartHomeActions {
    config: SmartHomeConfig,
    client: Client,
    api_token: Option<String>,
}

impl SmartHomeActions {
    pub async fn new(config: SmartHomeConfig) -> KlarnetResult<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_s))
            .build()
            .map_err(|e| KlarnetError::Action(e.to_string()))?;

        let api_token = std::env::var(&config.api_token_env).ok();

        Ok(Self {
            config,
            client,
            api_token,
        })
    }

    async fn call_home_assistant(
        &self,
        domain: &str,
        service: &str,
        data: Value,
    ) -> KlarnetResult<()> {
        let url = format!(
            "{}/api/services/{}/{}",
            self.config.api_url, domain, service
        );

        let mut request = self.client.post(&url).json(&data);

        if let Some(token) = &self.api_token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let response = request
            .send()
            .await
            .map_err(|e| KlarnetError::Action(format!("Home Assistant request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(KlarnetError::Action(format!(
                "Home Assistant error: {}",
                response.status()
            )));
        }

        Ok(())
    }
}

#[async_trait]
impl ActionHandler for SmartHomeActions {
    async fn can_handle(&self, action: &str) -> bool {
        action.starts_with("smart_home.")
    }

    async fn execute(&self, command: &LocalCommand) -> KlarnetResult<ActionResult> {
        match command.action.as_str() {
            "smart_home.lights" => {
                let state = command
                    .parameters
                    .get("state")
                    .and_then(|v| v.as_str())
                    .unwrap_or("on");

                let entity_id = command
                    .parameters
                    .get("room")
                    .and_then(|v| v.as_str())
                    .map(|room| format!("light.{}", room))
                    .unwrap_or_else(|| "light.all".to_string());

                let service = if state == "on" { "turn_on" } else { "turn_off" };

                self.call_home_assistant(
                    "light",
                    service,
                    json!({
                        "entity_id": entity_id
                    }),
                )
                .await?;

                Ok(ActionResult::success_with_message(format!(
                    "Свет {}",
                    if state == "on" {
                        "включен"
                    } else {
                        "выключен"
                    }
                )))
            }

            "smart_home.temperature" => {
                let temperature = command
                    .parameters
                    .get("temperature")
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| KlarnetError::Action("Temperature not provided".to_string()))?;

                let entity_id = command
                    .parameters
                    .get("room")
                    .and_then(|v| v.as_str())
                    .map(|room| format!("climate.{}", room))
                    .unwrap_or_else(|| "climate.main".to_string());

                self.call_home_assistant(
                    "climate",
                    "set_temperature",
                    json!({
                        "entity_id": entity_id,
                        "temperature": temperature
                    }),
                )
                .await?;

                Ok(ActionResult::success_with_message(format!(
                    "Температура установлена на {}°C",
                    temperature
                )))
            }

            _ => Err(KlarnetError::Action(format!(
                "Unknown smart home action: {}",
                command.action
            ))),
        }
    }

    fn name(&self) -> &str {
        "SmartHomeActions"
    }
}
