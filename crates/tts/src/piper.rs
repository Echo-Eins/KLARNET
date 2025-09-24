// crates/tts/src/piper.rs

use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use klarnet_core::{KlarnetError, KlarnetResult};
use tokio::process::Command;
use tokio::time;
use tracing::{debug, info};

use crate::{TtsBackend, TtsConfig};

pub struct PiperTts {
    config: TtsConfig,
}

impl PiperTts {
    pub async fn new(config: TtsConfig) -> KlarnetResult<Self> {
        if !config.runtime.piper_binary.exists() {
            info!(
                "Piper binary {} not found on disk; relying on PATH",
                config.runtime.piper_binary.display()
            );
        }
        Ok(Self { config })
    }
}

#[async_trait]
impl TtsBackend for PiperTts {
    async fn synthesize(&self, text: &str) -> KlarnetResult<Vec<u8>> {
        let mut command = Command::new(&self.config.runtime.piper_binary);
        command
            .arg("--model")
            .arg(&self.config.model)
            .arg("--output_raw")
            .arg("-")
            .arg("--input_text")
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if text.trim_start().starts_with('<') {
            command.arg("--ssml");
        }

        if let Some(device) = &self.config.device {
            if device.eq_ignore_ascii_case("cuda") || device.eq_ignore_ascii_case("gpu") {
                command.arg("--use_cuda");
            }
        }

        if let Some(language) = self.config.language.split('-').next() {
            if !language.is_empty() {
                command.arg("--language").arg(language);
            }
        }

        let mut child = command
            .spawn()
            .map_err(|err| KlarnetError::Action(format!("Failed to start Piper binary: {err}")))?;

        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| KlarnetError::Action("Failed to access Piper stdin".into()))?;

        let timeout = Duration::from_millis(self.config.runtime.request_timeout_ms);
        let mut payload = text.as_bytes().to_vec();
        if !text.ends_with('\n') {
            payload.push(b'\n');
        }

        time::timeout(timeout, async {
            use tokio::io::AsyncWriteExt;
            stdin.write_all(&payload).await.map_err(KlarnetError::Io)?;
            stdin.flush().await.map_err(KlarnetError::Io)?;
            Ok::<(), KlarnetError>(())
        })
            .await
            .map_err(|_| KlarnetError::Action("Timed out while sending text to Piper".into()))??;

        drop(stdin);

        let output = time::timeout(timeout, child.wait_with_output())
            .await
            .map_err(|_| KlarnetError::Action("Piper synthesis timed out".into()))?
            .map_err(|err| KlarnetError::Action(format!("Failed to wait for Piper: {err}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(KlarnetError::Action(format!(
                "Piper returned non-zero exit status: {}",
                stderr.trim()
            )));
        }

        if output.stdout.is_empty() {
            return Err(KlarnetError::Audio(
                "Piper produced empty PCM buffer".to_string(),
            ));
        }

        debug!(bytes = output.stdout.len(), "Received PCM data from Piper");

        Ok(output.stdout)
    }

    fn name(&self) -> &str {
        "Piper"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TtsRuntimeConfig;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[cfg(unix)]
    use std::fs::Permissions;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;

    fn mock_config(executable: &Path) -> TtsConfig {
        TtsConfig {
            runtime: TtsRuntimeConfig {
                python_path: PathBuf::from("python3"),
                silero_script: PathBuf::from("scripts/silero_tts.py"),
                piper_binary: executable.to_path_buf(),
                request_timeout_ms: 2_000,
            },
            ..TtsConfig::default()
        }
    }

    fn create_mock_binary(body: &str) -> (TempDir, PathBuf) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("mock-piper.py");
        fs::write(&path, body).unwrap();

        #[cfg(unix)]
        {
            fs::set_permissions(&path, Permissions::from_mode(0o755)).unwrap();
        }

        (dir, path)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn piper_succeeds_with_mock_binary() {
        let (dir, path) = create_mock_binary(
            r#"#!/usr/bin/env python3
import sys
text = sys.stdin.read()
if not text:
    sys.exit(1)
pcm = b"\x01\x00" * 12
sys.stdout.buffer.write(pcm)
sys.stdout.flush()
"#,
        );

        let config = mock_config(&path);
        let tts = PiperTts::new(config).await.unwrap();
        let pcm = tts.synthesize("hello").await.unwrap();
        assert!(!pcm.is_empty());
        drop(dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn piper_propagates_errors() {
        let (dir, path) = create_mock_binary(
            r#"#!/usr/bin/env python3
import sys
sys.stderr.write("boom\n")
sys.exit(2)
"#,
        );

        let config = mock_config(&path);
        let tts = PiperTts::new(config).await.unwrap();
        let err = tts.synthesize("fail").await.unwrap_err();
        assert!(matches!(err, KlarnetError::Action(msg) if msg.contains("non-zero")));
        drop(dir);
    }
}