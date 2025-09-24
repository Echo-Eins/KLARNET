// crates/observability/src/lib.rs

use opentelemetry_prometheus::PrometheusExporter;
use prometheus::{Encoder, TextEncoder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tracing::info;

pub mod tracing_config;
pub mod metrics;
pub mod health;

use metrics::{MetricType, Metrics};

/// Observability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    pub metrics_enabled: bool,
    pub traces_enabled: bool,
    pub prometheus_port: u16,
    pub export_interval_s: u64,
    pub service_name: String,
    pub traces_endpoint: Option<String>,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            metrics_enabled: true,
            traces_enabled: true,
            prometheus_port: 9090,
            export_interval_s: 10,
            service_name: "klarnet".to_string(),
            traces_endpoint: Some("http://localhost:4317".to_string()),
        }
    }
}

/// Metrics collector
pub struct MetricsCollector {
    config: ObservabilityConfig,
    metrics: Arc<Metrics>,
    exporter: Option<PrometheusExporter>,
    start_time: Instant,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self::with_config(ObservabilityConfig::default())
    }

    pub fn with_config(config: ObservabilityConfig) -> Self {
        let metrics = Arc::new(Metrics::new());

        let exporter = config.metrics_enabled.then(|| {
            opentelemetry_prometheus::exporter()
                .build()
                .expect("Failed to create Prometheus exporter")
        });

        if config.traces_enabled {
            Self::init_tracing(&config);
        }

        Self {
            config,
            metrics,
            exporter,
            start_time: Instant::now(),
        }
    }

    fn init_tracing(config: &ObservabilityConfig) {
        if let Some(endpoint) = &config.traces_endpoint {
            info!("Initializing OpenTelemetry tracing to {}", endpoint);
            // Initialize OTLP exporter
            // In production, configure proper OTLP exporter
        }
    }

    pub fn record(&self, metric: MetricType, value: f64) {
        self.metrics.record(metric, value);
    }

    pub fn add(&self, metric: MetricType, value: f64) {
        self.metrics.add(metric, value);
    }

    pub fn increment(&self, metric: MetricType) {
        self.metrics.increment(metric);
    }

    pub fn get_prometheus_metrics(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = prometheus::gather();
        let mut buffer = vec![];
        encoder.encode(&metric_families, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }

    pub fn get_health_status(&self) -> HealthStatus {
        HealthStatus {
            healthy: true,
            uptime_seconds: self.start_time.elapsed().as_secs(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            metrics: self.metrics.get_summary(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct HealthStatus {
    pub healthy: bool,
    pub uptime_seconds: u64,
    pub version: String,
    pub metrics: MetricsSummary,
}

#[derive(Debug, Serialize)]
pub struct MetricsSummary {
    pub total_requests: u64,
    pub audio_frames_processed: u64,
    pub transcriptions_completed: u64,
    pub actions_executed: u64,
    pub errors: u64,
}