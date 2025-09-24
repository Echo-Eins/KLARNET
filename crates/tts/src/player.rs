// crates/tts/src/player.rs

use std::time::Duration;

use klarnet_core::{KlarnetError, KlarnetResult};
use tracing::{debug, info, warn};

/// Simplified audio player that validates the desired output device and simulates
/// playback duration. In production this should stream audio via CPAL.

pub struct AudioPlayer {
    device_name: Option<String>,
}

impl AudioPlayer {
    pub fn new(preferred_device: Option<&str>) -> KlarnetResult<Self> {
        let device_name = preferred_device.map(|s| s.to_string());

        if let Some(name) = &device_name {
            info!("Using simulated audio output device: {}", name);
        } else {
            warn!("Audio output device not specified; playback will be simulated");
        }

        Ok(Self { device_name })
    }

    pub async fn play(&self, audio_data: &[u8], sample_rate: u32) -> KlarnetResult<()> {
        if audio_data.is_empty() {
            return Ok(());
        }

        if self.device_name.is_none() {
            debug!("Audio playback skipped due to missing output device");
            return Ok(());
        }

        // Play audio (simplified)
        if sample_rate == 0 {
            return Err(KlarnetError::Audio(
                "Invalid sample rate configured for playback".to_string(),
            ));
        }

        // In production, properly stream audio through cpal
        let sample_count = audio_data.len() / 2;
        let duration_ms = (sample_count as u64 * 1_000) / sample_rate as u64;

        info!(
            samples = sample_count,
            duration_ms,
            device = self.device_name.as_deref(),
            "Simulating audio playback"
        );

        tokio::time::sleep(Duration::from_millis(duration_ms)).await;

        Ok(())
    }
}