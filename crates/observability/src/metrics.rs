use crate::MetricsSummary;
use parking_lot::RwLock;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricType {
    TotalRequests,
    AudioFramesProcessed,
    TranscriptionsCompleted,
    ActionsExecuted,
    Errors,
    AudioPlaybackDurationMs,
    AudioPlaybackUnderruns,
    AudioPlaybackOverruns,
    AudioPlaybackRms,
    AudioCurrentVolume,
}

#[derive(Default)]
pub struct Metrics {
    values: RwLock<HashMap<MetricType, f64>>,
}

impl Metrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&self, metric: MetricType, value: f64) {
        let mut values = self.values.write();
        values.insert(metric, value);
    }

    pub fn add(&self, metric: MetricType, value: f64) {
        let mut values = self.values.write();
        *values.entry(metric).or_insert(0.0) += value;
    }

    pub fn increment(&self, metric: MetricType) {
        self.add(metric, 1.0);
    }

    pub fn get_summary(&self) -> MetricsSummary {
        let values = self.values.read();

        MetricsSummary {
            total_requests: values
                .get(&MetricType::TotalRequests)
                .copied()
                .unwrap_or_default() as u64,
            audio_frames_processed: values
                .get(&MetricType::AudioFramesProcessed)
                .copied()
                .unwrap_or_default() as u64,
            transcriptions_completed: values
                .get(&MetricType::TranscriptionsCompleted)
                .copied()
                .unwrap_or_default() as u64,
            actions_executed: values
                .get(&MetricType::ActionsExecuted)
                .copied()
                .unwrap_or_default() as u64,
            errors: values.get(&MetricType::Errors).copied().unwrap_or_default() as u64,
        }
    }

    pub fn get(&self, metric: MetricType) -> Option<f64> {
        let values = self.values.read();
        values.get(&metric).copied()
    }
}