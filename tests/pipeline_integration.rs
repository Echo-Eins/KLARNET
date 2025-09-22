use std::time::Duration;

use klarnet::{AudioPipeline, PipelineConfig};
use klarnet_core::AudioConfig;

#[tokio::test]
async fn pipeline_starts_and_stops() {
    let mut pipeline = AudioPipeline::new(
        PipelineConfig {
            frame_interval_ms: 5,
            speech_duration_ms: 50,
            inactivity_timeout_ms: 20,
            wake_word: "integration".to_string(),
        },
        AudioConfig {
            buffer_size: 8,
            ..AudioConfig::default()
        },
    );

    pipeline.start().await.expect("pipeline should start");
    tokio::time::sleep(Duration::from_millis(60)).await;
    pipeline.stop().await.expect("pipeline should stop");
}