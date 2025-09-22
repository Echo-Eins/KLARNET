// crates/tts/src/player.rs

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream};

pub struct AudioPlayer {
    device: Device,
}

impl AudioPlayer {
    pub fn new() -> KlarnetResult<Self> {
        let host = cpal::default_host();
        let device = host.default_output_device()
            .ok_or_else(|| KlarnetError::Audio("No output device available".to_string()))?;

        Ok(Self { device })
    }

    pub async fn play(&self, audio_data: &[u8]) -> KlarnetResult<()> {
        // Convert bytes to f32 samples
        let samples: Vec<f32> = audio_data
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
            .collect();

        // Play audio (simplified)
        info!("Playing {} samples", samples.len());

        // In production, properly stream audio through cpal
        tokio::time::sleep(std::time::Duration::from_millis(
            (samples.len() as u64 * 1000) / 48000
        )).await;

        Ok(())
    }
}