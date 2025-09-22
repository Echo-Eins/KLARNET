use std::time::Duration;

use klarnet::{AudioPipeline, PipelineConfig};
use klarnet_core::{AudioConfig, CommandType};
use nlu::NluMode;

#[tokio::test]
async fn wake_word_triggers_local_command() {
    let mut config = PipelineConfig::default();
    config.simulation.enabled = true;
    config.simulation.scripted_transcripts = vec!["Кларнет включи свет".to_string()];
    config.simulation.interval_ms = 10;
    config.nlu.mode = NluMode::Local;
    config.nlu.wake_words = vec!["кларнет".to_string()];

    let mut pipeline = AudioPipeline::new(config, AudioConfig::default());
    pipeline.start().await.expect("pipeline starts");

    let mut nlu_rx = pipeline
        .take_nlu_receiver()
        .expect("nlu receiver available");

    let result = tokio::time::timeout(Duration::from_secs(1), nlu_rx.recv())
        .await
        .expect("nlu result available")
        .expect("nlu result");

    assert!(result.wake_word_detected);
    assert!(matches!(result.command_type, CommandType::Local(_)));

    pipeline.stop().await.expect("pipeline stops");
}
