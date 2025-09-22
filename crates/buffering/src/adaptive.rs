// crates/buffering/src/adaptive.rs

/// Adaptive buffering strategy
pub struct AdaptiveBuffer {
    min_chunk_ms: usize,
    max_chunk_ms: usize,
    target_latency_ms: usize,
    current_chunk_ms: usize,
    latency_history: VecDeque<f32>,
    history_size: usize,
}

impl AdaptiveBuffer {
    pub fn new(min_chunk_ms: usize, max_chunk_ms: usize, target_latency_ms: usize) -> Self {
        Self {
            min_chunk_ms,
            max_chunk_ms,
            target_latency_ms,
            current_chunk_ms: (min_chunk_ms + max_chunk_ms) / 2,
            latency_history: VecDeque::with_capacity(10),
            history_size: 10,
        }
    }

    pub fn update_latency(&mut self, latency_ms: f32) {
        self.latency_history.push_back(latency_ms);
        if self.latency_history.len() > self.history_size {
            self.latency_history.pop_front();
        }

        self.adjust_chunk_size();
    }

    fn adjust_chunk_size(&mut self) {
        if self.latency_history.len() < 3 {
            return;
        }

        let avg_latency: f32 = self.latency_history.iter().sum::<f32>() / self.latency_history.len() as f32;

        if avg_latency > self.target_latency_ms as f32 * 1.2 {
            // Latency too high, reduce chunk size
            self.current_chunk_ms = (self.current_chunk_ms * 9 / 10).max(self.min_chunk_ms);
            debug!("Reducing chunk size to {}ms (latency: {:.1}ms)", self.current_chunk_ms, avg_latency);
        } else if avg_latency < self.target_latency_ms as f32 * 0.8 {
            // Latency low, can increase chunk size for efficiency
            self.current_chunk_ms = (self.current_chunk_ms * 11 / 10).min(self.max_chunk_ms);
            debug!("Increasing chunk size to {}ms (latency: {:.1}ms)", self.current_chunk_ms, avg_latency);
        }
    }

    pub fn get_optimal_chunk_size(&self) -> usize {
        self.current_chunk_ms
    }

    pub fn get_average_latency(&self) -> Option<f32> {
        if self.latency_history.is_empty() {
            None
        } else {
            Some(self.latency_history.iter().sum::<f32>() / self.latency_history.len() as f32)
        }
    }
}