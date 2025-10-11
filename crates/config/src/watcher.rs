use std::path::PathBuf;
use std::sync::Arc;

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use parking_lot::RwLock;
use tokio::sync::mpsc::UnboundedSender;
use tracing::{error, info};

use crate::{ConfigLoader, ConfigUpdateEvent, ConfigValidator, KlarnetConfig};
use klarnet_core::{KlarnetError, KlarnetResult};

pub struct ConfigWatcher {
    _watcher: RecommendedWatcher,
}

impl ConfigWatcher {
    pub fn start(
        path: PathBuf,
        config: Arc<RwLock<KlarnetConfig>>,
        tx: UnboundedSender<ConfigUpdateEvent>,
    ) -> KlarnetResult<Self> {
        let watch_path = path.clone();
        let tx_clone = tx.clone();
        let config_handle = Arc::clone(&config);

        let mut watcher =
            notify::recommended_watcher(move |event: Result<Event, notify::Error>| match event {
                Ok(event) => {
                    if !matches!(event.kind, EventKind::Modify(_)) {
                        return;
                    }

                    match ConfigLoader::load_from_file(&watch_path) {
                        Ok(new_config) => {
                            if let Err(err) = ConfigValidator::validate(&new_config) {
                                error!("Reloaded configuration is invalid: {err}");
                                return;
                            }

                            let mut guard = config_handle.write();
                            let old_config = guard.clone();
                            let changed_fields = super::detect_changes(&old_config, &new_config);
                            if changed_fields.is_empty() {
                                return;
                            }
                            *guard = new_config.clone();
                            drop(guard);

                            let event = ConfigUpdateEvent {
                                old_config,
                                new_config,
                                changed_fields,
                            };

                            if tx_clone.send(event).is_err() {
                                info!("No receivers listening for configuration updates");
                            }
                        }
                        Err(err) => {
                            error!("Failed to reload configuration: {err}");
                        }
                    }
                }
                Err(err) => error!("Configuration watcher error: {err}"),
            })
                .map_err(|e| KlarnetError::Config(format!("Failed to create watcher: {e}")))?;

        watcher
            .watch(&path, RecursiveMode::NonRecursive)
            .map_err(|e| KlarnetError::Config(format!("Failed to watch config {path:?}: {e}")))?;

        Ok(Self { _watcher: watcher })
    }
}
