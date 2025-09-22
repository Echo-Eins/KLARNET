use klarnet_core::{AudioConfig, KlarnetResult};

pub struct AudioProcessor {
    config: AudioConfig,
}

impl AudioProcessor {
    pub fn new(config: AudioConfig) -> Self {
        Self { config }
    }

    pub fn process(&mut self, input: &[f32]) -> KlarnetResult<Vec<f32>> {
        if self.config.channels > 1 {
            Ok(self.to_mono(input, self.config.channels as usize))
        } else {
            Ok(input.to_vec())
        }
    }

    fn to_mono(&self, input: &[f32], channels: usize) -> Vec<f32> {
        let frames = input.len() / channels;
        let mut mono = Vec::with_capacity(frames);

        for i in 0..frames {
            let mut sum = 0.0;
            for c in 0..channels {
                sum += input[i * channels + c];
            }
            mono.push(sum / channels as f32);
        }

        mono
    }

    pub fn normalize(&self, input: &[f32]) -> Vec<f32> {
        let max = input.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if max > 0.0 {
            input.iter().map(|s| s / max).collect()
        } else {
            input.to_vec()
        }
    }
}
