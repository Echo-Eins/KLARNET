use klarnet_core::{AudioFrame, KlarnetResult, VadEvent};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadConfig {
    pub frame_duration_ms: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            frame_duration_ms: 30,
        }
    }
}

pub struct VadProcessor {
    tx: mpsc::UnboundedSender<VadEvent>,
}

impl VadProcessor {
    pub fn new(_config: VadConfig) -> KlarnetResult<(Self, mpsc::UnboundedReceiver<VadEvent>)> {
        let (tx, rx) = mpsc::unbounded_channel();
        Ok((Self { tx }, rx))
    }

    pub async fn process_frame(&mut self, frame: AudioFrame) -> KlarnetResult<()> {
        let _ = self.tx.send(VadEvent::Silence {
            timestamp: frame.timestamp,
            duration: frame.duration,
        });
        Ok(())
    }
}
