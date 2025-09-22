// crates/api/src/lib.rs

use axum::{
    extract::{State, WebSocketUpgrade},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use klarnet_core::{AudioChunk, KlarnetResult, Transcript};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;
use tower_http::cors::CorsLayer;
use tracing::{error, info};

pub mod grpc;
pub mod websocket;
pub mod handlers;

use handlers::ApiHandlers;

/// API server configuration
#[derive(Debug, Clone)]
pub struct ApiConfig {
    pub host: String,
    pub port: u16,
    pub cors_enabled: bool,
    pub grpc_enabled: bool,
    pub grpc_port: u16,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 3000,
            cors_enabled: true,
            grpc_enabled: true,
            grpc_port: 50051,
        }
    }
}

/// API server
pub struct ApiServer {
    config: ApiConfig,
    handlers: Arc<ApiHandlers>,
}

impl ApiServer {
    pub async fn new(port: u16, metrics: Arc<klarnet_observability::MetricsCollector>) -> KlarnetResult<Self> {
        let config = ApiConfig {
            port,
            ..Default::default()
        };

        let handlers = Arc::new(ApiHandlers::new(metrics));

        Ok(Self { config, handlers })
    }

    pub async fn serve(self) -> KlarnetResult<()> {
        let app = self.create_router();

        let addr = format!("{}:{}", self.config.host, self.config.port)
            .parse()
            .map_err(|e| klarnet_core::KlarnetError::Network(format!("Invalid address: {}", e)))?;

        info!("API server listening on {}", addr);

        axum::Server::bind(&addr)
            .serve(app.into_make_service())
            .await
            .map_err(|e| klarnet_core::KlarnetError::Network(e.to_string()))?;

        Ok(())
    }

    fn create_router(&self) -> Router {
        let mut app = Router::new()
            .route("/health", get(health_check))
            .route("/ready", get(readiness_check))
            .route("/metrics", get(metrics_handler))
            .route("/stt/file", post(stt_file_handler))
            .route("/stt/stream", get(stt_stream_handler))
            .route("/nlu/interpret", post(nlu_handler))
            .route("/chat", post(chat_handler))
            .with_state(self.handlers.clone());

        if self.config.cors_enabled {
            app = app.layer(CorsLayer::permissive());
        }

        app
    }
}

// Health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now()
    }))
}

// Readiness check endpoint
async fn readiness_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "ready": true,
        "timestamp": chrono::Utc::now()
    }))
}

// Metrics endpoint
async fn metrics_handler(State(handlers): State<Arc<ApiHandlers>>) -> impl IntoResponse {
    handlers.get_metrics().await
}

// STT file upload endpoint
async fn stt_file_handler(
    State(handlers): State<Arc<ApiHandlers>>,
    body: bytes::Bytes,
) -> Result<Json<Transcript>, StatusCode> {
    handlers.transcribe_file(body).await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

// STT WebSocket streaming endpoint
async fn stt_stream_handler(
    ws: WebSocketUpgrade,
    State(handlers): State<Arc<ApiHandlers>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| websocket::handle_stt_stream(socket, handlers))
}

// NLU interpretation endpoint
async fn nlu_handler(
    State(handlers): State<Arc<ApiHandlers>>,
    Json(request): Json<NluRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    handlers.interpret_text(request.text).await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

#[derive(Deserialize)]
struct NluRequest {
    text: String,
}

// Chat endpoint for LLM integration
async fn chat_handler(
    State(handlers): State<Arc<ApiHandlers>>,
    Json(request): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, StatusCode> {
    handlers.chat(request).await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

#[derive(Deserialize)]
pub struct ChatRequest {
    pub message: String,
    pub context: Option<Vec<String>>,
}

#[derive(Serialize)]
pub struct ChatResponse {
    pub response: String,
    pub action: Option<String>,
}
