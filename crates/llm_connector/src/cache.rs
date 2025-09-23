use std::collections::HashMap;
use std::time::{Duration, Instant};

use parking_lot::RwLock;

use crate::CompletionResponse;

#[derive(Debug, Clone)]
pub struct LlmCacheEntry {
    pub value: CompletionResponse,
    pub expires_at: Instant,
}

#[derive(Debug)]
pub struct LlmCache {
    ttl: Duration,
    store: RwLock<HashMap<String, LlmCacheEntry>>,
}

impl LlmCache {
    pub fn new(ttl_s: u64) -> Self {
        Self {
            ttl: Duration::from_secs(ttl_s),
            store: RwLock::new(HashMap::new()),
        }
    }

    pub fn get(&self, key: &str) -> Option<CompletionResponse> {
        let mut guard = self.store.write();
        if let Some(entry) = guard.get(key) {
            if entry.expires_at > Instant::now() {
                return Some(entry.value.clone());
            }
        }
        guard.remove(key);
        None
    }

    pub fn set(&self, key: String, value: CompletionResponse) {
        let entry = LlmCacheEntry {
            value,
            expires_at: Instant::now() + self.ttl,
        };
        self.store.write().insert(key, entry);
    }

    pub fn invalidate_expired(&self) {
        let now = Instant::now();
        self.store.write().retain(|_, entry| entry.expires_at > now);
    }
}