use std::f32::consts::PI;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use klarnet_core::KlarnetResult;
use tts::{TtsBackend, TtsConfig, TtsEngine};

#[derive(Clone, Default)]
struct DummyBackend {
    state: Arc<Mutex<DummyBackendState>>,
    sample_rate: u32,
}

#[derive(Default)]
struct DummyBackendState {
    last_text: Option<String>,
}

impl DummyBackend {
    fn new(sample_rate: u32) -> Self {
        Self {
            state: Arc::new(Mutex::new(DummyBackendState::default())),
            sample_rate,
        }
    }

    fn last_text(&self) -> Option<String> {
        self.state.lock().unwrap().last_text.clone()
    }

    fn synthesize_wave(&self) -> Vec<u8> {
        let duration_s = 0.5f32;
        let total_samples = (self.sample_rate as f32 * duration_s) as usize;
        let mut pcm = Vec::with_capacity(total_samples * 2);
        for n in 0..total_samples {
            let angle = 2.0 * PI * 220.0 * (n as f32) / self.sample_rate as f32;
            let sample = (angle.sin() * i16::MAX as f32 * 0.2) as i16;
            pcm.extend_from_slice(&sample.to_le_bytes());
        }
        pcm
    }
}

#[async_trait]
impl TtsBackend for DummyBackend {
    async fn synthesize(&self, text: &str) -> KlarnetResult<Vec<u8>> {
        let mut state = self.state.lock().unwrap();
        state.last_text = Some(text.to_string());
        drop(state);
        Ok(self.synthesize_wave())
    }

    fn name(&self) -> &str {
        "dummy-test-backend"
    }
}

#[tokio::test]
async fn tts_engine_speak_announces_readiness() {
    let mut config = TtsConfig::default();
    config.cache.enabled = false;
    config.monitoring.enabled = false;

    let backend = DummyBackend::new(config.sample_rate);
    let engine = TtsEngine::from_backend(config, Box::new(backend.clone())).expect("engine");

    let phrase = "Кларнет готов к работе! Ожидание команды…";
    engine.speak(phrase).await.expect("speak completes");

    assert_eq!(backend.last_text().as_deref(), Some(phrase));

    let metrics = engine.metrics_snapshot_for_test();
    assert!(metrics.total_duration >= Duration::from_millis(100));
}