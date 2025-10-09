use crate::VadConfig;
pub(crate) struct Thresholds {
    pub speech: f32,
    pub release: f32,
}
impl Thresholds {
    pub fn new(noise_floor: f32, config: &VadConfig) -> Self {
        let speech = (noise_floor + config.energy_threshold).max(config.energy_threshold);
        let release = (speech * (1.0 - config.hysteresis_ratio)).max(noise_floor);
        Self { speech, release }
    }
}
