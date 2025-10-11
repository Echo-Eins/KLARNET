use klarnet_core::{AudioConfig, KlarnetResult};
#[derive(Debug, Clone)]
pub struct AudioProcessor {
    config: AudioConfig,
}

impl AudioProcessor {
    pub fn new(config: AudioConfig) -> Self {
        Self { config }
    }

    pub fn process(&self, input: &[f32], input_channels: u16) -> KlarnetResult<Vec<f32>> {
        let target_channels = self.config.channels.max(1) as usize;
        let input_channels = input_channels.max(1) as usize;

        if input_channels == target_channels {
            return Ok(input.to_vec());
        }

        if target_channels == 1 {
            return Ok(self.to_mono(input, input_channels));
        }

        if input_channels == 1 {
            return Ok(self.upmix(input, target_channels));
        }

        // When both channel counts differ and are greater than one, downmix to mono
        // before upmixing to the desired configuration. This keeps processing simple
        // while ensuring we always emit the requested number of channels.
        let mono = self.to_mono(input, input_channels);
        Ok(self.upmix(&mono, target_channels))
    }

    fn to_mono(&self, input: &[f32], channels: usize) -> Vec<f32> {
        let frames = input.len() / channels;
        let mut mono = Vec::with_capacity(frames);

        for frame in input.chunks_exact(channels) {
            let sum: f32 = frame.iter().sum();
            mono.push(sum / channels as f32);
        }

        mono
    }
    
    fn upmix(&self, input: &[f32], channels: usize) -> Vec<f32> {
        let mut output = Vec::with_capacity(input.len() * channels);
        for &sample in input {
            for _ in 0..channels {
                output.push(sample);
            }
        }
        output
    }

    pub fn target_channels(&self) -> u16 {
        self.config.channels.max(1)
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
