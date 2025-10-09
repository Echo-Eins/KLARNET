use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket};
use futures::StreamExt;
use serde_json::json;
use tokio::time::Instant;
use tracing::{error, info};

use crate::handlers::{build_chunk_from_pcm, decode_pcm_s16le, ApiHandlers};

const SAMPLE_RATE: u32 = 16_000;

pub async fn handle_stt_stream(mut socket: WebSocket, handlers: Arc<ApiHandlers>) {
    let session_id = handlers.create_session();

    info!(%session_id, "Starting STT WebSocket session");

    if socket
        .send(Message::Text(
            json!({
                "type": "welcome",
                "session_id": session_id,
                "sample_rate": SAMPLE_RATE,
                "format": "pcm_s16le"
            })
            .to_string(),
        ))
        .await
        .is_err()
    {
        return;
    }

    while let Some(message) = socket.next().await {
        match message {
            Ok(Message::Binary(data)) => match decode_pcm_s16le(&data) {
                Ok(samples) => {
                    if let Some(chunk_pcm) = handlers.append_session_audio(&session_id, &samples) {
                        if let Some(whisper) = handlers.whisper_engine() {
                            let started = Instant::now();
                            let chunk = build_chunk_from_pcm(&chunk_pcm, SAMPLE_RATE);
                            let mut engine = whisper.lock().await;
                            match engine.transcribe(chunk).await {
                                Ok(transcript) => {
                                    let response = json!({
                                        "type": "transcript",
                                        "text": transcript.full_text,
                                        "latency_ms": started.elapsed().as_millis() as u64,
                                    });
                                    if socket
                                        .send(Message::Text(response.to_string()))
                                        .await
                                        .is_err()
                                    {
                                        break;
                                    }
                                }
                                Err(err) => {
                                    error!(%session_id, "Streaming transcription failed: {err}");
                                    let payload = json!({
                                        "type": "error",
                                        "message": err.to_string(),
                                    });
                                    let _ = socket.send(Message::Text(payload.to_string())).await;
                                }
                            }
                        }
                    }
                }
                Err(err) => {
                    error!(%session_id, "Failed to decode audio payload: {err}");
                    let payload = json!({"type": "error", "message": err.to_string()});
                    if socket
                        .send(Message::Text(payload.to_string()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            },
            Ok(Message::Text(text)) => {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                    if value["type"] == "ping" {
                        let payload = json!({"type": "pong"});
                        let _ = socket.send(Message::Text(payload.to_string())).await;
                    }
                }
            }
            Ok(Message::Close(_)) => {
                info!(%session_id, "WebSocket client closed session");
                break;
            }
            Err(err) => {
                error!(%session_id, "WebSocket error: {err}");
                break;
            }
            _ => {}
        }
    }

    handlers.cleanup_sessions(Duration::from_secs(300));
    info!(%session_id, "STT WebSocket session terminated");
}
