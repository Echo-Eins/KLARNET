// crates/vad/src/energy.rs
pub struct EnergyVad {
    config: VadConfig,
    energy_history: Vec<f32>,
    history_size: usize,
}

impl EnergyVad {
    pub fn new(config: VadConfig) -> Self {
        Self {
            config,
            energy_history: Vec::with_capacity(100),
            history_size: 100,
        }
    }

    fn calculate_adaptive_threshold(&self) -> f32 {
        if self.energy_history.is_empty() {
            return self.config.energy_threshold;
        }

        let mean: f32 = self.energy_history.iter().sum::<f32>() / self.energy_history.len() as f32;
        let variance: f32 = self.energy_history.iter()
            .map(|e| (e - mean).powi(2))
            .sum::<f32>() / self.energy_history.len() as f32;
        let std_dev = variance.sqrt();

        mean + std_dev * 1.5
    }
}

#[async_trait]
impl VoiceActivityDetector for EnergyVad {
    async fn process(&mut self, frame: AudioFrame) -> KlarnetResult<Option<VadEvent>> {
        let energy = frame.data.iter().map(|s| s * s).sum::<f32>() / frame.data.len() as f32;

        self.energy_history.push(energy);
        if self.energy_history.len() > self.history_size {
            self.energy_history.remove(0);
        }

        let threshold = self.calculate_adaptive_threshold();

        let event = if energy > threshold {
            Some(VadEvent::SpeechFrame {
                timestamp: frame.timestamp,
                pcm: frame.data.clone(),
                energy,
            })
        } else {
            Some(VadEvent::Silence {
                timestamp: frame.timestamp,
                duration: Duration::from_millis(self.config.frame_duration_ms as u64),
            })
        };

        Ok(event)
    }

    fn reset(&mut self) {
        self.energy_history.clear();
    }
}