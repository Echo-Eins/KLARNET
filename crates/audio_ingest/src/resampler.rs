#[derive(Debug, Clone, Default)]
pub struct Resampler {
    target_rate: u32,
}

impl Resampler {
    pub fn new(target_rate: u32) -> Self {
        Self { target_rate }
    }

    pub fn target_rate(&self) -> u32 {
        self.target_rate
    }

    pub fn resample(&self, input: &[f32], source_rate: u32) -> Vec<f32> {
        if source_rate == 0 || self.target_rate == 0 || source_rate == self.target_rate {
            return input.to_vec();
        }

        let ratio = self.target_rate as f32 / source_rate as f32;
        let output_len = ((input.len() as f32) * ratio).round().max(1.0) as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_pos = i as f32 / ratio;
            let idx = src_pos.floor() as usize;
            let frac = src_pos - idx as f32;

            if idx + 1 >= input.len() {
                output.push(input[input.len().saturating_sub(1)]);
            } else {
                let a = input[idx];
                let b = input[idx + 1];
                output.push(a + (b - a) * frac);
            }
        }

        output
    }
}
