use std::sync::Arc;
use std::time::{Duration, Instant};

use llm_connector::{LlmConnector, LlmMetricsSnapshot, LlmProviderKind, Usage};
use tokio::sync::{Mutex as AsyncMutex, OwnedSemaphorePermit, Semaphore};

use crate::LlmModeConfig;

/// Runtime helper that encapsulates concurrency and usage tracking for LLM calls.
pub(crate) struct LlmRuntime {
    pub(crate) connector: Arc<LlmConnector>,
    semaphore: Arc<Semaphore>,
    min_interval: Option<Duration>,
    last_call: AsyncMutex<Option<Instant>>,
    last_usage: AsyncMutex<Option<LlmUsageRecord>>,
    config: LlmModeConfig,
}

impl LlmRuntime {
    pub(crate) fn new(
        connector: Arc<LlmConnector>,
        semaphore: Arc<Semaphore>,
        min_interval: Option<Duration>,
        config: LlmModeConfig,
    ) -> Self {
        Self {
            connector,
            semaphore,
            min_interval,
            last_call: AsyncMutex::new(None),
            last_usage: AsyncMutex::new(None),
            config,
        }
    }

    pub(crate) async fn acquire(&self) -> OwnedSemaphorePermit {
        let permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("LLM semaphore closed");

        if let Some(interval) = self.min_interval {
            let mut last_call = self.last_call.lock().await;
            if let Some(last) = *last_call {
                let elapsed = last.elapsed();
                if elapsed < interval {
                    tokio::time::sleep(interval - elapsed).await;
                }
            }
            *last_call = Some(Instant::now());
        }

        permit
    }

    pub(crate) async fn record_usage(&self, usage: Usage, latency: Duration) {
        let config = self.connector.config();
        let provider = match &config.provider {
            LlmProviderKind::OpenRouter => "openrouter".to_string(),
            LlmProviderKind::DeepSeek => "deepseek".to_string(),
            LlmProviderKind::OpenAI => "openai".to_string(),
            LlmProviderKind::Custom(name) => name.clone(),
        };

        let mut guard = self.last_usage.lock().await;
        *guard = Some(LlmUsageRecord {
            usage,
            latency,
            provider,
            model: config.model.clone(),
        });
    }
}

pub(crate) async fn take_usage(&self) -> Option<LlmUsageRecord> {
    let mut guard = self.last_usage.lock().await;
    guard.take()
}

pub(crate) fn metrics_snapshot(&self) -> LlmMetricsSnapshot {
    self.connector.metrics_snapshot()
}

pub(crate) fn summary(&self) -> LlmConfigurationSummary {
    LlmConfigurationSummary {
        provider: self.config.provider.clone(),
        model: self.config.model.clone(),
        cache_enabled: self.config.cache_enabled,
        max_concurrent_requests: self.config.max_concurrent_requests,
        min_request_interval_ms: self.config.min_request_interval_ms,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LlmUsageRecord {
    pub usage: Usage,
    pub latency: Duration,
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Clone)]
pub struct LlmConfigurationSummary {
    pub provider: String,
    pub model: String,
    pub cache_enabled: bool,
    pub max_concurrent_requests: usize,
    pub min_request_interval_ms: u64,
}

#[derive(Debug)]
pub struct LlmInterpretation {
    pub intent_name: Option<String>,
    pub confidence: f32,
    pub parameters: serde_json::Map<String, serde_json::Value>,
    pub entities: Vec<klarnet_core::Entity>,
    pub action: Option<String>,
    pub route: Option<String>,
    pub response_text: Option<String>,
    pub function_name: Option<String>,
    pub usage: Usage,
}
