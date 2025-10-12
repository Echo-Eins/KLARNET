use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

fn default_history_size() -> usize {
    5
}

fn default_recovery_rate() -> f32 {
    0.3
}

fn default_min_weight() -> f32 {
    0.1
}

/// Configuration for acknowledgement phrase selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AckConfig {
    /// List of acknowledgement phrases that can be spoken after the wake word.
    pub phrases: Vec<String>,
    /// Sliding window size used to avoid immediate repetitions.
    #[serde(default = "default_history_size")]
    pub history_size: usize,
    /// Speed at which a phrase recovers its selection weight.
    #[serde(default = "default_recovery_rate")]
    pub recovery_rate: f32,
    /// Minimum weight value applied during decay.
    #[serde(default = "default_min_weight")]
    pub min_weight: f32,
}

impl Default for AckConfig {
    fn default() -> Self {
        Self {
            phrases: vec![
                "Слушаю".to_string(),
                "Да".to_string(),
                "Готов".to_string(),
                "Я вас слушаю".to_string(),
                "Готова к работе".to_string(),
                "Какая команда?".to_string(),
                "Внимательно слушаю".to_string(),
            ],
            history_size: default_history_size(),
            recovery_rate: default_recovery_rate(),
            min_weight: default_min_weight(),
        }
    }
}

/// Weighted random selector with soft repetition avoidance.
pub struct AckSelector {
    config: AckConfig,
    weights: HashMap<String, f32>,
    history: Vec<String>,
}

impl AckSelector {
    /// Create a new selector using the provided configuration.
    pub fn new(config: AckConfig) -> Self {
        let mut weights = HashMap::new();
        for phrase in &config.phrases {
            weights.insert(phrase.clone(), 1.0);
        }

        Self {
            config,
            weights,
            history: Vec::new(),
        }
    }

    /// Select the next acknowledgement phrase according to the dynamic weights.
    pub fn select_next(&mut self) -> Option<String> {
        if self.config.phrases.is_empty() {
            return None;
        }

        let mut rng = rand::thread_rng();
        let total_weight: f32 = self.weights.values().sum();
        if total_weight <= f32::EPSILON {
            return self.config.phrases.first().cloned();
        }
        let mut roll = rng.gen::<f32>() * total_weight;

        let mut selected = self.config.phrases[0].clone();
        for phrase in &self.config.phrases {
            let weight = self.weights.get(phrase).copied().unwrap_or(1.0);
            if roll <= weight {
                selected = phrase.clone();
                break;
            }
            roll -= weight;
        }

        if self.config.phrases.len() > 1 {
            if let Some(last) = self.history.last() {
                if last == &selected {
                    if let Some(alternative) = self
                        .config
                        .phrases
                        .iter()
                        .filter(|phrase| *phrase != &selected)
                        .max_by(|a, b| {
                            let aw = self.weights.get(*a).copied().unwrap_or(1.0);
                            let bw = self.weights.get(*b).copied().unwrap_or(1.0);
                            aw.partial_cmp(&bw).unwrap()
                        })
                    {
                        selected = alternative.clone();
                    }
                }
            }
        }

        self.update_after_selection(&selected);
        Some(selected)
    }

    /// Return current weights for debugging or observability.
    pub fn weights(&self) -> Vec<(String, f32)> {
        let mut weights: Vec<_> = self
            .weights
            .iter()
            .map(|(phrase, weight)| (phrase.clone(), *weight))
            .collect();
        weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        weights
    }

    /// Return the current history window.
    pub fn history(&self) -> &[String] {
        &self.history
    }

    fn update_after_selection(&mut self, selected: &str) {
        self.history.push(selected.to_string());
        if self.history.len() > self.config.history_size {
            if let Some(oldest) = self.history.first().cloned() {
                self.increase_weight(&oldest, self.config.recovery_rate);
            }
            self.history.remove(0);
        }

        self.decrease_weight(selected, 0.5);

        let recovery_phrases: Vec<String> = self
            .config
            .phrases
            .iter()
            .filter(|phrase| *phrase != selected)
            .cloned()
            .collect();
        for phrase in recovery_phrases {
            self.increase_weight(&phrase, self.config.recovery_rate * 0.5);
        }
    }

    fn decrease_weight(&mut self, phrase: &str, factor: f32) {
        let current = self.weights.get(phrase).copied().unwrap_or(1.0);
        let new_weight = (current * (1.0 - factor)).max(self.config.min_weight);
        self.weights.insert(phrase.to_string(), new_weight);
    }

    fn increase_weight(&mut self, phrase: &str, amount: f32) {
        let current = self.weights.get(phrase).copied().unwrap_or(1.0);
        let new_weight = (current + amount).min(1.0);
        self.weights.insert(phrase.to_string(), new_weight);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn selector_avoids_immediate_repeats() {
        let config = AckConfig {
            phrases: vec!["A".into(), "B".into(), "C".into()],
            history_size: 2,
            recovery_rate: 0.3,
            min_weight: 0.1,
        };
        let mut selector = AckSelector::new(config);

        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut last = String::new();
        let mut repeats = 0;

        for _ in 0..100 {
            let phrase = selector.select_next().expect("phrase");
            *counts.entry(phrase.clone()).or_default() += 1;
            if phrase == last {
                repeats += 1;
            }
            last = phrase;
        }

        assert!(repeats < 5, "too many immediate repeats: {}", repeats);
        for (phrase, count) in counts {
            assert!(
                count > 20 && count < 50,
                "phrase {} count {}",
                phrase,
                count
            );
        }
    }

    #[test]
    fn weights_recover_over_time() {
        let mut selector = AckSelector::new(AckConfig::default());
        let phrase = selector.config.phrases[0].clone();
        let initial = selector.weights.get(&phrase).copied().unwrap();

        selector.update_after_selection(&phrase);
        let after_use = selector.weights.get(&phrase).copied().unwrap();
        assert!(after_use < initial);

        for _ in 0..10 {
            selector.select_next();
        }

        let recovered = selector.weights.get(&phrase).copied().unwrap();
        assert!(recovered > after_use);
    }
}