// crates/tts/src/silero.rs

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use std::ffi::OsString;

use async_trait::async_trait;
use klarnet_core::{
    resolve_project_path, venv_site_packages_directories, KlarnetError, KlarnetResult,
};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio::time;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::{TtsBackend, TtsConfig};

/// Silero TTS backend communicating with a Python helper process over stdio.

pub struct SileroTts {
    config: TtsConfig,
    script_path: PathBuf,
    model_path: PathBuf,
    process: Arc<Mutex<Option<SileroProcess>>>,
}

fn candidate_file_names(model: &str) -> Vec<OsString> {
    let mut names = Vec::new();
    let spec_path = PathBuf::from(model);

    if let Some(file_name) = spec_path.file_name() {
        names.push(file_name.to_os_string());
    } else if !model.is_empty() {
        names.push(OsString::from(model));
    }

    if !model.ends_with(".pt") {
        if let Some(file_name) = spec_path.file_name() {
            let mut with_ext = file_name.to_os_string();
            with_ext.push(".pt");
            if !names.contains(&with_ext) {
                names.push(with_ext);
            }
        } else if !model.is_empty() {
            let mut with_ext = OsString::from(model);
            with_ext.push(".pt");
            names.push(with_ext);
        }
    }

    names
}

fn resolve_silero_model_path(model: &str) -> KlarnetResult<PathBuf> {
    let trimmed = model.trim();
    if trimmed.is_empty() {
        return Err(KlarnetError::Action(
            "Silero model identifier must not be empty".to_string(),
        ));
    }

    let search_names = candidate_file_names(trimmed);
    let mut candidates = Vec::new();

    let direct_path = resolve_project_path(trimmed);
    if direct_path.exists() {
        candidates.push((direct_path, "config"));
    }

    let models_dir = resolve_project_path("models/silero");
    if models_dir.is_dir() {
        for name in &search_names {
            let candidate = models_dir.join(name);
            if candidate.exists() {
                candidates.push((candidate, "models/silero"));
            }
        }
    }

    for site_packages in venv_site_packages_directories() {
        let silero_dir = site_packages.join("silero");
        if !silero_dir.is_dir() {
            continue;
        }

        for name in &search_names {
            let candidate = silero_dir.join(name);
            if candidate.exists() {
                candidates.push((candidate, "venv"));
            }
        }
    }

    let mut unique = Vec::new();
    for (path, source) in candidates {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
        if unique.iter().any(|(existing, _)| existing == &canonical) {
            continue;
        }
        unique.push((canonical, (path, source)));
    }

    if unique.is_empty() {
        let mut message = format!(
            "Unable to locate Silero model '{}' in the project or virtual environment",
            model
        );
        if models_dir.is_dir() {
            message.push_str(&format!("; expected file inside {}", models_dir.display()));
        }
        return Err(KlarnetError::Action(message));
    }

    if unique.len() > 1 {
        let details = unique
            .iter()
            .map(|(canonical, (original, source))| {
                if canonical == original {
                    format!("{} [{source}]", canonical.display())
                } else {
                    format!(
                        "{} (canonical {}) [{source}]",
                        original.display(),
                        canonical.display()
                    )
                }
            })
            .collect::<Vec<_>>()
            .join(", ");

        return Err(KlarnetError::Action(format!(
            "Multiple Silero models matching '{}' found: {}. Remove duplicates or set 'tts.model' to an explicit path.",
            model, details
        )));
    }

    let (canonical, (original, _)) = unique.into_iter().next().unwrap();
    if !canonical.is_file() {
        return Err(KlarnetError::Action(format!(
            "Resolved Silero model '{}' is not a file",
            original.display()
        )));
    }

    Ok(canonical)
}

impl SileroTts {
    pub async fn new(config: TtsConfig) -> KlarnetResult<Self> {
        let script_path = resolve_project_path(&config.runtime.silero_script);

        if !script_path.exists() {
            return Err(KlarnetError::Action(format!(
                "Silero script not found: {}",
                script_path.display()
            )));
        }

        let model_path = resolve_silero_model_path(&config.model)?;

        info!(model = %model_path.display(), "Resolved Silero model path");

        let process = Self::spawn_process(&config, &script_path, &model_path).await?;

        Ok(Self {
            config,
            script_path,
            model_path,
            process: Arc::new(Mutex::new(Some(process))),
        })
    }

    async fn spawn_process(
        config: &TtsConfig,
        script_path: &Path,
        model_path: &Path,
    ) -> KlarnetResult<SileroProcess> {
        let python_path = if config.runtime.python_path.is_relative()
            && config.runtime.python_path.components().count() > 1
        {
            resolve_project_path(&config.runtime.python_path)
        } else {
            config.runtime.python_path.clone()
        };

        let mut command = Command::new(&python_path);
        command
            .arg("-u")
            .arg(script_path)
            .arg("--model")
            .arg(model_path)
            .arg("--speaker")
            .arg(&config.speaker)
            .arg("--sample-rate")
            .arg(config.sample_rate.to_string())
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        if let Some(device) = &config.device {
            command.arg("--device").arg(device);
        }

        if python_path.is_absolute() && !python_path.is_file() {
            info!(
                "Python executable {} not found on disk; relying on PATH",
                python_path.display()
            );
        }

        let mut child = command.spawn().map_err(|err| {
            KlarnetError::Action(format!("Failed to start Silero runtime: {err}"))
        })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| KlarnetError::Action("Unable to capture Silero stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| KlarnetError::Action("Unable to capture Silero stdout".into()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| KlarnetError::Action("Unable to capture Silero stderr".into()))?;

        info!("Spawned Silero Python helper process");

        Ok(SileroProcess {
            child,
            stdin: BufWriter::new(stdin),
            stdout: BufReader::new(stdout),
            stderr: BufReader::new(stderr),
        })
    }

    async fn ensure_process(
        &self,
    ) -> KlarnetResult<tokio::sync::MutexGuard<'_, Option<SileroProcess>>> {
        let mut guard = self.process.lock().await;
        let mut restart_required = false;

        if let Some(process) = guard.as_mut() {
            match process.child.try_wait() {
                Ok(Some(status)) => {
                    warn!(
                        exit = ?status,
                        "Silero helper exited; restarting before next synthesis"
                    );
                    restart_required = true;
                }
                Ok(None) => {}
                Err(err) => {
                    return Err(KlarnetError::Action(format!(
                        "Failed to poll Silero process status: {err}"
                    )));
                }
            }
        } else {
            restart_required = true;
        }

        if restart_required {
            let process =
                Self::spawn_process(&self.config, &self.script_path, &self.model_path).await?;
            *guard = Some(process);
        }

        Ok(guard)
    }
}

#[async_trait]
impl TtsBackend for SileroTts {
    async fn synthesize(&self, text: &str) -> KlarnetResult<Vec<u8>> {
        let mut guard = self.ensure_process().await?;
        let timeout = Duration::from_millis(self.config.runtime.request_timeout_ms);
        let request = SileroRequest {
            id: Uuid::new_v4().to_string(),
            text: text.to_string(),
            language: self.config.language.clone(),
            speaker: self.config.speaker.clone(),
            sample_rate: self.config.sample_rate,
            speed: self.config.speed,
            model: self.model_path.to_string_lossy().to_string(),
            device: self.config.device.clone(),
        };

        let synthesis_result = {
            let process = guard
                .as_mut()
                .expect("Silero process should be available after ensure_process");
            process.synthesize(request, timeout).await
        };

        match synthesis_result {
            Ok(result) => {
                debug!(
                    response_id = %result.id,
                    quality = ?result.quality,
                    duration_ms = result.duration_ms,
                    "Received PCM from Silero"
                );
                Ok(result.pcm)
            }
            Err(err) => {
                error!("Silero synthesis failed: {err}");
                *guard = None; // force restart on next attempt
                Err(err)
            }
        }
    }

    fn name(&self) -> &str {
        "Silero"
    }
}

#[derive(Debug, Serialize)]
struct SileroRequest {
    id: String,
    text: String,
    language: String,
    speaker: String,
    sample_rate: u32,
    speed: f32,
    model: String,
    device: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SileroResponse {
    id: String,
    status: SileroResponseStatus,
    pcm_len: Option<usize>,
    message: Option<String>,
    quality: Option<f32>,
    duration_ms: Option<u64>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum SileroResponseStatus {
    Ok,
    Error,
}

struct SileroSynthesis {
    id: String,
    pcm: Vec<u8>,
    quality: Option<f32>,
    duration_ms: Option<u64>,
}

struct SileroProcess {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    stderr: BufReader<ChildStderr>,
}

impl SileroProcess {
    async fn synthesize(
        &mut self,
        request: SileroRequest,
        timeout: Duration,
    ) -> KlarnetResult<SileroSynthesis> {
        let payload = serde_json::to_vec(&request)?;
        self.stdin.write_all(&payload).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;

        let mut header_line = String::new();
        let bytes_read = time::timeout(timeout, self.stdout.read_line(&mut header_line))
            .await
            .map_err(|_| {
                KlarnetError::Action("Timed out waiting for Silero response header".into())
            })??;

        if bytes_read == 0 {
            return Err(self
                .collect_process_failure("Silero process closed stdout unexpectedly")
                .await);
        }

        let response: SileroResponse = serde_json::from_str(header_line.trim()).map_err(|err| {
            KlarnetError::Action(format!("Failed to parse Silero response header: {err}"))
        })?;

        match response.status {
            SileroResponseStatus::Ok => {
                let pcm_len = response.pcm_len.ok_or_else(|| {
                    KlarnetError::Action(
                        "Silero response missing pcm_len for successful synthesis".into(),
                    )
                })?;

                let mut pcm = vec![0u8; pcm_len];
                time::timeout(timeout, self.stdout.read_exact(&mut pcm))
                    .await
                    .map_err(|_| {
                        KlarnetError::Action("Timed out while reading PCM data from Silero".into())
                    })??;

                Ok(SileroSynthesis {
                    id: response.id,
                    pcm,
                    quality: response.quality,
                    duration_ms: response.duration_ms,
                })
            }
            SileroResponseStatus::Error => Err(KlarnetError::Action(
                response
                    .message
                    .unwrap_or_else(|| "Silero returned an error".to_string()),
            )),
        }
    }

    async fn collect_process_failure(&mut self, context: &str) -> KlarnetError {
        let mut stderr = String::new();
        let _ = time::timeout(
            Duration::from_millis(200),
            self.stderr.read_to_string(&mut stderr),
        )
        .await;
        if stderr.trim().is_empty() {
            KlarnetError::Action(context.to_string())
        } else {
            KlarnetError::Action(format!("{context}: {}", stderr.trim()))
        }
    }
}

impl Drop for SileroProcess {
    fn drop(&mut self) {
        if let Err(err) = self.child.start_kill() {
            if err.kind() != std::io::ErrorKind::InvalidInput {
                error!("Failed to terminate Silero process: {err}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TtsRuntimeConfig;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    fn mock_config(script_path: &Path, model_path: &Path) -> TtsConfig {
        let mut config = TtsConfig {
            runtime: TtsRuntimeConfig {
                python_path: PathBuf::from("python3"),
                silero_script: script_path.to_path_buf(),
                piper_binary: PathBuf::from("piper"),
                request_timeout_ms: 2_000,
            },
            ..TtsConfig::default()
        };
        config.model = model_path.to_string_lossy().to_string();
        config
    }

    fn write_mock_script(temp: &mut NamedTempFile, body: &str) {
        use std::io::Write;
        writeln!(temp, "{body}").unwrap();
        temp.flush().unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn silero_succeeds_with_mock_process() {
        let mut script = NamedTempFile::new().unwrap();
        write_mock_script(
            &mut script,
            r#"import sys, json
for line in sys.stdin:
    if not line.strip():
        continue
    req = json.loads(line)
    pcm = b"\x01\x00" * 10
    header = {"id": req["id"], "status": "ok", "pcm_len": len(pcm), "quality": 0.9, "duration_ms": 42}
    sys.stdout.write(json.dumps(header) + "\n")
    sys.stdout.flush()
    sys.stdout.buffer.write(pcm)
    sys.stdout.flush()
"#,
        );

        let model = NamedTempFile::new().unwrap();
        let config = mock_config(script.path(), model.path());
        let tts = SileroTts::new(config).await.unwrap();
        let pcm = tts.synthesize("hello").await.unwrap();
        assert!(!pcm.is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn silero_reports_error_from_process() {
        let mut script = NamedTempFile::new().unwrap();
        write_mock_script(
            &mut script,
            r#"import sys, json
for line in sys.stdin:
    if not line.strip():
        continue
    req = json.loads(line)
    header = {"id": req["id"], "status": "error", "message": "backend failure"}
    sys.stdout.write(json.dumps(header) + "\n")
    sys.stdout.flush()
"#,
        );

        let model = NamedTempFile::new().unwrap();
        let config = mock_config(script.path(), model.path());
        let tts = SileroTts::new(config).await.unwrap();
        let err = tts.synthesize("boom").await.unwrap_err();
        match err {
            KlarnetError::Action(message) => assert!(message.contains("backend failure")),
            other => panic!("Unexpected error type: {other:?}"),
        }
    }
}
