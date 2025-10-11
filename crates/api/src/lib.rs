use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use bytes::Bytes;
use klarnet_core::{KlarnetError, KlarnetResult, Transcript};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tracing::{error, info};

pub mod grpc;
pub mod handlers;
pub mod websocket;

use handlers::ApiHandlers;

#[derive(Clone)]
struct ApiState {
    handlers: Arc<ApiHandlers>,
    sample_rate: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub cors_origins: Vec<String>,
    #[serde(default = "default_grpc_enabled")]
    pub grpc_enabled: bool,
    #[serde(default = "default_grpc_port")]
    pub grpc_port: u16,
}

fn default_enabled() -> bool {
    true
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    3000
}

fn default_grpc_enabled() -> bool {
    true
}

fn default_grpc_port() -> u16 {
    50_051
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
            host: default_host(),
            port: default_port(),
            cors_origins: vec!["*".to_string()],
            grpc_enabled: default_grpc_enabled(),
            grpc_port: default_grpc_port(),
        }
    }
}

pub struct ApiServer {
    config: ApiConfig,
    handlers: Arc<ApiHandlers>,
    sample_rate: u32,
}

impl ApiServer {
    pub fn new(config: ApiConfig, handlers: ApiHandlers, sample_rate: u32) -> Self {
        Self {
            config,
            handlers: Arc::new(handlers),
            sample_rate,
        }
    }

    pub fn with_shared_handlers(
        config: ApiConfig,
        handlers: Arc<ApiHandlers>,
        sample_rate: u32,
    ) -> Self {
        Self {
            config,
            handlers,
            sample_rate,
        }
    }

    pub async fn serve(
        self,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> KlarnetResult<Option<JoinHandle<()>>> {
        if !self.config.enabled {
            info!("API server is disabled");
            return Ok(None);
        }

        let addr: SocketAddr = format!("{}:{}", self.config.host, self.config.port)
            .parse()
            .map_err(|err| KlarnetError::Network(format!("Invalid API address: {err}")))?;

        let router = self.build_router();
        info!(%addr, "HTTP API listening");

        let listener = TcpListener::bind(addr)
            .await
            .map_err(|err| KlarnetError::Network(format!("Failed to bind API listener: {err}")))?;

        let http_handle = tokio::spawn(async move {
            if let Err(err) = axum::serve(listener, router.into_make_service())
                .with_graceful_shutdown(shutdown)
                .await
            {
                error!("API server terminated with error: {err}");
            }
        });

        if self.config.grpc_enabled {
            let grpc_addr: SocketAddr = format!("{}:{}", self.config.host, self.config.grpc_port)
                .parse()
                .map_err(|err| KlarnetError::Network(format!("Invalid gRPC address: {err}")))?;
            let grpc_service = grpc::GrpcService::new(self.handlers.clone(), self.sample_rate);
            tokio::spawn(async move {
                if let Err(err) = grpc_service.serve(grpc_addr).await {
                    error!("gRPC server terminated with error: {err}");
                }
            });
        }

        Ok(Some(http_handle))
    }

    fn build_router(&self) -> Router {
        let state = ApiState {
            handlers: self.handlers.clone(),
            sample_rate: self.sample_rate,
        };

        let mut router = Router::new()
            .route("/health", get(health_check))
            .route("/ready", get(readiness_check))
            .route("/metrics", get(metrics_handler))
            .route("/stt/file", post(stt_file_handler))
            .route("/stt/stream", get(stt_stream_handler))
            .route("/nlu/interpret", post(nlu_handler))
            .route("/chat", post(chat_handler))
            .with_state(state);

        if self.config.cors_origins.iter().any(|origin| origin == "*") {
            router = router.layer(CorsLayer::permissive());
        }

        router
    }
}

async fn health_check() -> impl IntoResponse {
    Json(json!({ "status": "healthy" }))
}

async fn readiness_check() -> impl IntoResponse {
    Json(json!({ "ready": true }))
}

async fn metrics_handler(State(state): State<ApiState>) -> impl IntoResponse {
    state.handlers.metrics_snapshot()
}

async fn stt_file_handler(
    State(state): State<ApiState>,
    body: Bytes,
) -> Result<Json<Transcript>, Response> {
    state
        .handlers
        .transcribe_file(body, state.sample_rate)
        .await
        .map(Json)
        .map_err(|err| error_response(StatusCode::INTERNAL_SERVER_ERROR, err))
}

async fn stt_stream_handler(
    ws: axum::extract::WebSocketUpgrade,
    State(state): State<ApiState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| websocket::handle_stt_stream(socket, state.handlers))
}

async fn nlu_handler(
    State(state): State<ApiState>,
    Json(payload): Json<NluRequest>,
) -> Result<Json<serde_json::Value>, Response> {
    state
        .handlers
        .interpret_text(payload.text)
        .await
        .map(Json)
        .map_err(|err| error_response(StatusCode::INTERNAL_SERVER_ERROR, err))
}

async fn chat_handler(
    State(state): State<ApiState>,
    Json(request): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, Response> {
    state
        .handlers
        .chat(request)
        .await
        .map(Json)
        .map_err(|err| error_response(StatusCode::INTERNAL_SERVER_ERROR, err))
}

fn error_response(status: StatusCode, err: KlarnetError) -> Response {
    error!("API request failed: {err}");
    (status, Json(json!({ "error": err.to_string() }))).into_response()
}

#[derive(Deserialize)]
struct NluRequest {
    text: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub response: String,
    pub action: Option<String>,
}
