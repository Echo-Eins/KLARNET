// crates/audio_ingest/src/processor.rs
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

pub struct AudioProcessor {
    config: AudioConfig,
    resampler: Option<Box<dyn Resampler<f32>>>,
}

impl AudioProcessor {
    pub fn new(config: AudioConfig) -> Self {
        Self {
            config,
            resampler: None,
        }
    }

    pub fn init_resampler(&mut self, from_rate: u32, to_rate: u32) -> KlarnetResult<()> {
        if from_rate == to_rate {
            self.resampler = None;
            return Ok(());
        }

        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };

        let resampler = SincFixedIn::<f32>::new(
            to_rate as f64 / from_rate as f64,
            2.0,
            params,
            1024,
            self.config.channels as usize,
        ).map_err(|e| KlarnetError::Audio(format!("Failed to create resampler: {}", e)))?;

        self.resampler = Some(Box::new(resampler));
        Ok(())
    }

    pub fn process(&mut self, input: &[f32]) -> KlarnetResult<Vec<f32>> {
        // Convert to mono if needed
        let mono = if self.config.channels > 1 {
            self.to_mono(input, self.config.channels as usize)
        } else {
            input.to_vec()
        };

        // Resample if needed
        if let Some(resampler) = &mut self.resampler {
            let mut output = vec![vec![0f32; resampler.output_frames_max()]; 1];
            let input_frames = vec![mono];

            let (_, output_len) = resampler.process_into_buffer(&input_frames, &mut output, None)
                .map_err(|e| KlarnetError::Audio(format!("Resampling failed: {}", e)))?;

            Ok(output[0][..output_len].to_vec())
        } else {
            Ok(mono)
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