// crates/tts/src/silero.rs

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use klarnet_core::{KlarnetError, KlarnetResult};
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
    process: Arc<Mutex<Option<SileroProcess>>>,
}

impl SileroTts {
    pub async fn new(config: TtsConfig) -> KlarnetResult<Self> {
        if !config.runtime.silero_script.exists() {
            return Err(KlarnetError::Action(format!(
                "Silero script not found: {}",
                config.runtime.silero_script.display()
            )));
        }

        let process = Self::spawn_process(&config).await?;

        Ok(Self {
            config,
            process: Arc::new(Mutex::new(Some(process))),
        })
    }

    async fn spawn_process(config: &TtsConfig) -> KlarnetResult<SileroProcess> {
        let mut command = Command::new(&config.runtime.python_path);
        command
            .arg("-u")
            .arg(&config.runtime.silero_script)
            .arg("--model")
            .arg(&config.model)
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

        if !Path::new(&config.runtime.python_path).is_file() {
            info!(
                "Python executable {} not found on disk; relying on PATH",
                config.runtime.python_path.display()
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
            let process = Self::spawn_process(&self.config).await?;
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
            model: self.config.model.clone(),
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

    fn mock_config(script_path: &Path) -> TtsConfig {
        TtsConfig {
            runtime: TtsRuntimeConfig {
                python_path: PathBuf::from("python3"),
                silero_script: script_path.to_path_buf(),
                piper_binary: PathBuf::from("piper"),
                request_timeout_ms: 2_000,
            },
            ..TtsConfig::default()
        }
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

        let config = mock_config(script.path());
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

        let config = mock_config(script.path());
        let tts = SileroTts::new(config).await.unwrap();
        let err = tts.synthesize("boom").await.unwrap_err();
        match err {
            KlarnetError::Action(message) => assert!(message.contains("backend failure")),
            other => panic!("Unexpected error type: {other:?}"),
        }
    }
}