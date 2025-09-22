// crates/vad/src/hybrid.rs
pub struct HybridVad {
    webrtc: WebRtcVad,
    energy: EnergyVad,
    config: VadConfig,
}

impl HybridVad {
    pub fn new(config: VadConfig) -> KlarnetResult<Self> {
        Ok(Self {
            webrtc: WebRtcVad::new(config.clone())?,
            energy: EnergyVad::new(config.clone()),
            config,
        })
    }
}

#[async_trait]
impl VoiceActivityDetector for HybridVad {
    async fn process(&mut self, frame: AudioFrame) -> KlarnetResult<Option<VadEvent>> {
        let webrtc_result = self.webrtc.process(frame.clone()).await?;
        let energy_result = self.energy.process(frame.clone()).await?;

        // Combine results with weighted voting
        match (webrtc_result, energy_result) {
            (Some(VadEvent::SpeechFrame { .. }), Some(VadEvent::SpeechFrame { .. })) => {
                Ok(Some(VadEvent::SpeechFrame {
                    timestamp: frame.timestamp,
                    pcm: frame.data.clone(),
                    energy: 0.0,
                }))
            }
            (Some(VadEvent::Silence { .. }), Some(VadEvent::Silence { .. })) => {
                Ok(Some(VadEvent::Silence {
                    timestamp: frame.timestamp,
                    duration: Duration::from_millis(self.config.frame_duration_ms as u64),
                }))
            }
            _ => {
                // Mixed results - use WebRTC as primary
                Ok(webrtc_result)
            }
        }
    }

    fn reset(&mut self) {
        self.webrtc.reset();
        self.energy.reset();
    }
}
