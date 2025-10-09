pub(crate) fn frame_energy(pcm: &[f32]) -> f32 {
    if pcm.is_empty() {
        return 0.0;
    }
    pcm.iter().map(|sample| sample * sample).sum::<f32>() / pcm.len() as f32
}
pub(crate) fn update_noise_floor(noise_floor: f32, energy: f32, rate: f32) -> f32 {
    let clamped_rate = rate.clamp(0.0, 1.0);
    let updated = (1.0 - clamped_rate) * noise_floor + clamped_rate * energy;
    updated.max(super::EPSILON)
}
