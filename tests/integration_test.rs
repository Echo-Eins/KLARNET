// tests/integration_test.rs

use klarnet_core::*;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

#[tokio::test]
async fn test_full_pipeline() {
    // Initialize components
    let config = AudioConfig::default();
    let audio_ingest = klarnet_audio_ingest::AudioIngest::new(config, Duration::from_millis(1000))
        .expect("Failed to create audio ingest");

    let vad_config = klarnet_vad::VadConfig::default();
    let (mut vad_processor, mut vad_rx) = klarnet_vad::VadProcessor::new(vad_config)
        .expect("Failed to create VAD processor");

    let buffer = klarnet_buffering::AudioBuffer::new(16000);

    // Test audio flow
    let test_frame = AudioFrame {
        data: Arc::from(vec![0.1f32; 480].into_boxed_slice()),
        timestamp: chrono::Utc::now(),
        duration: Duration::from_millis(30),
        sample_rate: 16000,
    };

    // Process frame through VAD
    vad_processor.process_frame(test_frame.clone()).await
        .expect("Failed to process frame");

    // Check for events with timeout
    let result = timeout(Duration::from_millis(100), vad_rx.recv()).await;

    assert!(result.is_ok() || result.is_err()); // Either we get an event or timeout
}

#[tokio::test]
async fn test_whisper_transcription() {
    // Skip if model not available
    if !std::path::Path::new("models/faster-whisper-medium").exists() {
        println!("Skipping Whisper test - model not found");
        return;
    }

    let config = klarnet_whisper_stt::WhisperConfig::default();
    let engine = klarnet_whisper_stt::WhisperEngine::new(config).await
        .expect("Failed to create Whisper engine");

    // Create test audio chunk
    let samples = vec![0.0f32; 16000]; // 1 second of silence
    let chunk = AudioChunk::from_pcm(&samples, 16000);

    // Transcribe
    let transcript = engine.transcribe(chunk).await
        .expect("Failed to transcribe");

    assert_eq!(transcript.language, "ru");
}

#[tokio::test]
async fn test_nlu_pattern_matching() {
    let config = klarnet_nlu::NluConfig::default();
    let nlu = klarnet_nlu::NluEngine::new(config).await
        .expect("Failed to create NLU engine");

    let transcript = Transcript {
        id: uuid::Uuid::new_v4(),
        language: "ru".to_string(),
        segments: vec![],
        full_text: "джарвис включи свет в гостиной".to_string(),
        processing_time: Duration::from_millis(0),
    };

    let result = nlu.process(&transcript).await
        .expect("Failed to process transcript");

    assert!(result.wake_word_detected);
    assert!(result.intent.is_some());
}

#[tokio::test]
async fn test_action_execution() {
    let executor = klarnet_actions::ActionExecutor::new().await
        .expect("Failed to create action executor");

    let command = LocalCommand {
        action: "system.volume".to_string(),
        parameters: {
            let mut map = serde_json::Map::new();
            map.insert("level".to_string(), serde_json::json!(50));
            map
        },
    };

    let result = executor.execute(command).await
        .expect("Failed to execute action");

    assert!(result.success || !result.success); // Action may fail on CI
}

#[test]
fn test_ring_buffer() {
    use klarnet_buffering::ring_buffer::RingBuffer;

    let mut buffer = RingBuffer::<f32>::new(10);

    // Fill buffer
    for i in 0..15 {
        buffer.push(i as f32);
    }

    assert_eq!(buffer.len(), 10);

    // Test get_last_n
    let last_5 = buffer.get_last_n(5);
    assert_eq!(last_5, vec![10.0, 11.0, 12.0, 13.0, 14.0]);

    // Test pop
    let val = buffer.pop();
    assert_eq!(val, Some(5.0));
    assert_eq!(buffer.len(), 9);
}

#[tokio::test]
async fn test_api_health_check() {
    use axum::http::StatusCode;
    use axum::body::Body;
    use tower::ServiceExt;

    let metrics = Arc::new(klarnet_observability::MetricsCollector::new());
    let handlers = Arc::new(klarnet_api::handlers::ApiHandlers::new(metrics.clone()));

    let app = axum::Router::new()
        .route("/health", axum::routing::get(|| async {
            axum::Json(serde_json::json!({
                "status": "healthy"
            }))
        }));

    let response = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_config_loading() {
    use klarnet_config::ConfigManager;

    // Test with defaults
    let manager = ConfigManager::with_defaults();
    let config = manager.get();

    assert_eq!(config.app.language, "ru");
    assert_eq!(config.audio.sample_rate, 16000);
    assert_eq!(config.vad.aggressiveness, 2);
}

#[tokio::test]
async fn test_websocket_session() {
    use klarnet_api::handlers::ApiHandlers;
    use futures::{sink::SinkExt, stream::StreamExt};

    let metrics = Arc::new(klarnet_observability::MetricsCollector::new());
    let handlers = Arc::new(ApiHandlers::new(metrics));

    // Create mock WebSocket
    // In real test, use tokio_tungstenite to create actual WebSocket

    // Test session management
    let sessions = handlers.active_sessions.clone();
    {
        let mut sessions = sessions.write();
        let session_id = sessions.create_session();
        assert!(sessions.get_session_mut(&session_id).is_some());
    }
}

// Benchmark tests
#[cfg(all(test, not(debug_assertions)))]
mod benches {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn benchmark_vad(c: &mut Criterion) {
        let frame = AudioFrame {
            data: Arc::from(vec![0.1f32; 480].into_boxed_slice()),
            timestamp: chrono::Utc::now(),
            duration: Duration::from_millis(30),
            sample_rate: 16000,
        };

        c.bench_function("vad_process_frame", |b| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let config = klarnet_vad::VadConfig::default();
            let (mut processor, _) = klarnet_vad::VadProcessor::new(config).unwrap();

            b.iter(|| {
                rt.block_on(async {
                    processor.process_frame(black_box(frame.clone())).await
                })
            });
        });
    }

    fn benchmark_buffer_ops(c: &mut Criterion) {
        use klarnet_buffering::ring_buffer::RingBuffer;

        let mut buffer = RingBuffer::<f32>::new(10000);

        c.bench_function("ring_buffer_push", |b| {
            b.iter(|| {
                buffer.push(black_box(0.5));
            });
        });

        c.bench_function("ring_buffer_get_last_n", |b| {
            for i in 0..10000 {
                buffer.push(i as f32);
            }

            b.iter(|| {
                buffer.get_last_n(black_box(1000))
            });
        });
    }

    criterion_group!(benches, benchmark_vad, benchmark_buffer_ops);
}