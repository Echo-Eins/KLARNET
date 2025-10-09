use crate::VadConfig;
pub(crate) fn tune_speech_threshold(threshold: f32) -> f32 {
    threshold
}
pub(crate) fn tune_release_threshold(threshold: f32) -> f32 {
    threshold
}
pub(crate) fn supports_webrtc(_config: &VadConfig) -> bool {
    false
}
