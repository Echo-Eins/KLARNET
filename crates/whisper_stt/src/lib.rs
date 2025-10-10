mod config;
mod streaming;
use std::fs as stdfs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use klarnet_core::{
    resolve_project_path, venv_site_packages_directories, AudioChunk, KlarnetError, KlarnetResult,
    Transcript, TranscriptSegment, WordInfo,
};
use serde::Deserialize;
use tokio::fs;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::time::{sleep, timeout};
use tracing::debug;

use config::WhisperBackendConfig as BackendConfig;
pub use config::{
    WhisperBackendConfig, WhisperConfig, WhisperMetrics, WhisperModelConfig, WhisperPythonConfig,
    SUPPORTED_LANGUAGES,
};
pub use streaming::StreamingWhisper;

type BackendHandle = Box<dyn WhisperBackend + Send>;

pub struct WhisperEngine {
    config: WhisperConfig,
    backend: BackendHandle,
    metrics: WhisperMetrics,
}

impl WhisperEngine {
    pub async fn new(config: WhisperConfig) -> KlarnetResult<Self> {
        config.validate()?;

        let mut backend: BackendHandle = match config.backend.clone() {
            BackendConfig::Python(python) => {
                Box::new(PythonWhisperProcess::new(python)) as BackendHandle
            }
        };

        timeout(config.initialization_timeout(), backend.initialize(&config))
            .await
            .map_err(|_| {
                KlarnetError::Stt("Whisper backend initialization timed out".to_string())
            })??;

        Ok(Self {
            config,
            backend,
            metrics: WhisperMetrics::default(),
        })
    }

    #[cfg(test)]
    async fn with_backend(
        config: WhisperConfig,
        mut backend: BackendHandle,
    ) -> KlarnetResult<Self> {
        config.validate()?;
        backend.initialize(&config).await?;
        Ok(Self {
            config,
            backend,
            metrics: WhisperMetrics::default(),
        })
    }

    pub async fn transcribe(&mut self, chunk: AudioChunk) -> KlarnetResult<Transcript> {
        let pcm = chunk.to_pcm();
        let mut attempt = 0usize;

        loop {
            let start = Instant::now();
            let response = timeout(
                self.config.request_timeout(),
                self.backend.transcribe(&chunk, &pcm, &self.config),
            )
            .await;

            match response {
                Ok(Ok(result)) => {
                    let elapsed = start.elapsed();
                    let transcript = self.build_transcript(&chunk, result, elapsed);
                    self.metrics.total_processed += 1;
                    self.metrics.last_processing_time = Some(elapsed);
                    self.metrics.cumulative_processing_time += elapsed;
                    self.metrics.total_retries += attempt as u64;
                    return Ok(transcript);
                }
                Ok(Err(err)) => {
                    self.metrics.total_errors += 1;
                    if attempt >= self.config.retry_attempts {
                        self.metrics.total_retries += attempt as u64;
                        return Err(err);
                    }
                    attempt += 1;
                    self.restart_backend().await?;
                    sleep(self.config.retry_backoff()).await;
                }
                Err(_) => {
                    self.metrics.total_timeouts += 1;
                    if attempt >= self.config.retry_attempts {
                        self.metrics.total_retries += attempt as u64;
                        return Err(KlarnetError::Stt(
                            "Whisper transcription timed out".to_string(),
                        ));
                    }
                    attempt += 1;
                    self.restart_backend().await?;
                    sleep(self.config.retry_backoff()).await;
                }
            }
        }
    }

    pub fn get_metrics(&self) -> WhisperMetrics {
        self.metrics.clone()
    }

    pub fn shutdown(&mut self) {
        self.backend.shutdown();
    }

    async fn restart_backend(&mut self) -> KlarnetResult<()> {
        self.metrics.total_restarts += 1;
        self.backend.restart(&self.config).await.map_err(|err| {
            self.metrics.total_errors += 1;
            err
        })
    }

    fn build_transcript(
        &self,
        chunk: &AudioChunk,
        response: WhisperResponse,
        elapsed: Duration,
    ) -> Transcript {
        let mut segments: Vec<TranscriptSegment> = Vec::with_capacity(response.segments.len());

        for segment in response.segments {
            let words: Vec<WordInfo> = segment
                .words
                .into_iter()
                .map(WhisperWord::into_word_info)
                .collect();

            let confidence = segment.confidence.unwrap_or_else(|| {
                if words.is_empty() {
                    0.0
                } else {
                    words.iter().map(|w| w.confidence).sum::<f32>() / words.len() as f32
                }
            });

            segments.push(TranscriptSegment {
                start: segment.start,
                end: segment.end,
                text: segment.text,
                confidence,
                words,
            });
        }

        let full_text = segments
            .iter()
            .map(|segment| segment.text.trim())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

        Transcript {
            id: chunk.id,
            language: response
                .language
                .unwrap_or_else(|| self.config.language.clone()),
            segments,
            full_text,
            processing_time: elapsed,
        }
    }
}

#[async_trait]
trait WhisperBackend {
    async fn initialize(&mut self, config: &WhisperConfig) -> KlarnetResult<()>;
    async fn transcribe(
        &mut self,
        chunk: &AudioChunk,
        pcm: &[f32],
        config: &WhisperConfig,
    ) -> KlarnetResult<WhisperResponse>;
    async fn restart(&mut self, config: &WhisperConfig) -> KlarnetResult<()> {
        self.shutdown();
        self.initialize(config).await
    }
    fn shutdown(&mut self);
}

struct PythonWhisperProcess {
    config: WhisperPythonConfig,
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    stdout: Option<BufReader<ChildStdout>>,
}

impl PythonWhisperProcess {
    fn new(config: WhisperPythonConfig) -> Self {
        Self {
            config,
            child: None,
            stdin: None,
            stdout: None,
        }
    }

    async fn spawn_child(&mut self, config: &WhisperConfig) -> KlarnetResult<()> {
        if let Some(cache_dir) = config.model.cache_dir.as_ref() {
            fs::create_dir_all(cache_dir)
                .await
                .map_err(|err| KlarnetError::Stt(err.to_string()))?;
        }

        let model_path = resolve_whisper_model_path(&config.model.model_path)?;

        debug!(
            "Resolved Whisper model path for faster-whisper backend: {}",
            model_path.display(),
        );

        let mut command = Command::new(&self.config.executable);
        command
            .arg("-u")
            .arg(&self.config.script)
            .arg("--model-path")
            .arg(model_path.to_string_lossy().to_string())
            .arg("--language")
            .arg(&config.language)
            .arg("--compute-type")
            .arg(&config.model.compute_type)
            .arg("--device")
            .arg(&config.model.device)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit());

        for extra in &self.config.extra_args {
            command.arg(extra);
        }

        if let Some(cache_dir) = config.model.cache_dir.as_ref() {
            command.env("CT2_CACHE_DIR", cache_dir);
        }

        for (key, value) in &self.config.env {
            command.env(key, value);
        }

        let mut child = command
            .spawn()
            .map_err(|err| KlarnetError::Stt(format!("Failed to spawn Whisper process: {err}")))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| KlarnetError::Stt("Whisper process stdin unavailable".to_string()))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| KlarnetError::Stt("Whisper process stdout unavailable".to_string()))?;

        self.stdin = Some(stdin);
        self.stdout = Some(BufReader::new(stdout));
        self.child = Some(child);
        Ok(())
    }

    async fn ensure_running(&mut self, config: &WhisperConfig) -> KlarnetResult<()> {
        let should_restart = if let Some(child) = self.child.as_mut() {
            match child.try_wait() {
                Ok(Some(_)) => true,
                Ok(None) => false,
                Err(err) => {
                    return Err(KlarnetError::Stt(format!(
                        "Failed to poll Whisper process: {err}"
                    )))
                }
            }
        } else {
            true
        };

        if should_restart {
            self.shutdown();
            self.spawn_child(config).await?;
        }

        Ok(())
    }
}

fn resolve_whisper_model_path(spec: &Path) -> KlarnetResult<PathBuf> {
    #[derive(Clone)]
    struct Candidate {
        canonical: PathBuf,
        original: PathBuf,
        source: String,
    }

    fn push_candidate(candidates: &mut Vec<Candidate>, path: PathBuf, source: &str) {
        if !path.exists() {
            return;
        }

        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
        if candidates
            .iter()
            .any(|existing| existing.canonical == canonical)
        {
            return;
        }

        candidates.push(Candidate {
            canonical,
            original: path,
            source: source.to_string(),
        });
    }

    fn detect_ct2_weights(dir: &Path, display: &Path) -> KlarnetResult<bool> {
        let has_weights = stdfs::read_dir(dir)
            .map_err(|err| {
                KlarnetError::Stt(format!(
                    "Failed to inspect Whisper model directory '{}': {err}",
                    display.display()
                ))
            })?
            .filter_map(Result::ok)
            .any(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .map(|name| name.starts_with("model.bin"))
                    .unwrap_or(false)
            });

        Ok(has_weights)
    }

    let mut candidates = Vec::new();
    let resolved = resolve_project_path(spec);
    if resolved.exists() {
        push_candidate(&mut candidates, resolved, "config");
    }

    if let Some(root) = klarnet_core::detect_project_root() {
        if !spec.is_absolute() {
            push_candidate(&mut candidates, root.join(spec), "project root");
        }

        let models_dir = root.join("models");
        if models_dir.is_dir() {
            if let Some(file_name) = spec.file_name() {
                push_candidate(
                    &mut candidates,
                    models_dir.join(file_name),
                    "project models",
                );
            }
        }

        let release_dir = root.join("target").join("release");
        if release_dir.is_dir() {
            if !spec.is_absolute() {
                push_candidate(&mut candidates, release_dir.join(spec), "target/release");
            }

            if let Some(file_name) = spec.file_name() {
                push_candidate(
                    &mut candidates,
                    release_dir.join("models").join(file_name),
                    "target/release models",
                );
            }
        }
    }

    if !spec.is_absolute() {
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                push_candidate(&mut candidates, exe_dir.join(spec), "binary directory");

                if let Some(file_name) = spec.file_name() {
                    push_candidate(
                        &mut candidates,
                        exe_dir.join("models").join(file_name),
                        "binary models",
                    );
                }
            }
        }
    }

    if let Some(file_name) = spec.file_name() {
        for site_packages in venv_site_packages_directories() {
            push_candidate(&mut candidates, site_packages.join(file_name), "venv");
        }
    }

    if candidates.is_empty() {
        let message = if let Some(name) = spec.file_name() {
            format!(
                "Unable to locate Whisper model directory '{}' in the project or virtual environment",
                name.to_string_lossy()
            )
        } else {
            format!(
                "Unable to locate Whisper model directory at '{}'",
                spec.display()
            )
        };
        return Err(KlarnetError::Stt(message));
    }

    let mut directories = Vec::new();
    let mut non_directories = Vec::new();

    for candidate in candidates {
        if candidate.canonical.is_dir() {
            directories.push(candidate);
        } else {
            non_directories.push(candidate);
        }
    }

    if directories.is_empty() {
        if let Some(candidate) = non_directories.into_iter().next() {
            return Err(KlarnetError::Stt(format!(
                "Resolved Whisper model path '{}' is not a directory",
                candidate.original.display()
            )));
        }

        return Err(KlarnetError::Stt(format!(
            "Unable to locate Whisper model directory at '{}'",
            spec.display()
        )));
    }

    let mut with_weights = Vec::new();
    let mut without_weights = Vec::new();

    for candidate in directories {
        if detect_ct2_weights(&candidate.canonical, &candidate.original)? {
            with_weights.push(candidate);
        } else {
            without_weights.push(candidate);
        }
    }

    if with_weights.len() == 1 {
        return Ok(with_weights.remove(0).canonical);
    }

    if with_weights.len() > 1 {
        let details = with_weights
            .iter()
            .map(|candidate| {
                if candidate.canonical == candidate.original {
                    format!("{} [{}]", candidate.canonical.display(), candidate.source)
                } else {
                    format!(
                        "{} (canonical {}) [{}]",
                        candidate.original.display(),
                        candidate.canonical.display(),
                        candidate.source
                    )
                }
            })
            .collect::<Vec<_>>()
            .join(", ");

        return Err(KlarnetError::Stt(format!(
            "Multiple Whisper models matching '{}' contain CTranslate2 weights: {}. Set 'stt.model.model_path' to the desired directory.",
            spec.display(),
            details
        )));
    }

    let details = without_weights
        .iter()
        .map(|candidate| format!("{} [{}]", candidate.original.display(), candidate.source))
        .collect::<Vec<_>>()
        .join(", ");

    Err(KlarnetError::Stt(format!(
        "Whisper model directory '{}' does not contain CTranslate2 weights (expected files named 'model.bin*'). Checked: {}. Download a CTranslate2-converted Whisper model, e.g. 'medium' from https://huggingface.co/Systran/faster-whisper-medium.",
        spec.display(),
        details
    )))
}

#[async_trait]
impl WhisperBackend for PythonWhisperProcess {
    async fn initialize(&mut self, config: &WhisperConfig) -> KlarnetResult<()> {
        self.spawn_child(config).await
    }

    async fn transcribe(
        &mut self,
        _chunk: &AudioChunk,
        pcm: &[f32],
        config: &WhisperConfig,
    ) -> KlarnetResult<WhisperResponse> {
        self.ensure_running(config).await?;

        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| KlarnetError::Stt("Whisper stdin not available".to_string()))?;
        let stdout = self
            .stdout
            .as_mut()
            .ok_or_else(|| KlarnetError::Stt("Whisper stdout not available".to_string()))?;

        let sample_count = pcm.len() as u32;
        stdin
            .write_all(&sample_count.to_le_bytes())
            .await
            .map_err(|err| KlarnetError::Stt(format!("Failed to write sample count: {err}")))?;

        let mut buffer = Vec::with_capacity(pcm.len() * std::mem::size_of::<f32>());
        for sample in pcm {
            buffer.extend_from_slice(&sample.to_le_bytes());
        }

        stdin
            .write_all(&buffer)
            .await
            .map_err(|err| KlarnetError::Stt(format!("Failed to write PCM payload: {err}")))?;
        stdin
            .flush()
            .await
            .map_err(|err| KlarnetError::Stt(format!("Failed to flush Whisper stdin: {err}")))?;

        let mut response = String::new();
        let read = stdout
            .read_line(&mut response)
            .await
            .map_err(|err| KlarnetError::Stt(format!("Failed to read Whisper response: {err}")))?;

        if read == 0 {
            return Err(KlarnetError::Stt(
                "Whisper process closed stdout".to_string(),
            ));
        }

        let trimmed = response.trim();
        if trimmed.is_empty() {
            return Err(KlarnetError::Stt(
                "Whisper process returned empty response".to_string(),
            ));
        }

        let parsed: WhisperResponse = serde_json::from_str(trimmed)?;
        Ok(parsed)
    }

    fn shutdown(&mut self) {
        self.stdin.take();
        self.stdout.take();

        if let Some(mut child) = self.child.take() {
            let _ = child.start_kill();
        }
    }
}

impl Drop for PythonWhisperProcess {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[derive(Debug, Deserialize)]
struct WhisperResponse {
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    segments: Vec<WhisperSegment>,
}

#[derive(Debug, Deserialize)]
struct WhisperSegment {
    start: f64,
    end: f64,
    text: String,
    #[serde(default)]
    confidence: Option<f32>,
    #[serde(default)]
    words: Vec<WhisperWord>,
}

#[derive(Debug, Deserialize)]
struct WhisperWord {
    word: String,
    start: f64,
    end: f64,
    #[serde(default)]
    confidence: Option<f32>,
    #[serde(default)]
    probability: Option<f32>,
}

impl WhisperWord {
    fn confidence_value(&self) -> f32 {
        self.confidence
            .or(self.probability)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0)
    }

    fn into_word_info(self) -> WordInfo {
        let confidence = self.confidence_value();
        WordInfo {
            word: self.word,
            start: self.start,
            end: self.end,
            confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex as StdMutex, OnceLock};
    use std::time::Duration;

    use super::*;
    use chrono::Utc;
    use klarnet_core::{AudioConfig, AudioFrame};
    use tempfile::tempdir;
    use tokio::sync::Mutex;

    fn env_lock() -> &'static StdMutex<()> {
        static LOCK: OnceLock<StdMutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| StdMutex::new(()))
    }

    const PYTHON_TEST_TRANSCRIPT: &str = "привет от python";
    const PYTHON_BACKEND_TEMPLATE: &str = r#"
import argparse
import json
import struct
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--model-path")
parser.add_argument("--language")
parser.add_argument("--compute-type")
parser.add_argument("--device")
parser.parse_args()


def read_exact(count):
    data = b""
    while len(data) < count:
        chunk = sys.stdin.buffer.read(count - len(data))
        if not chunk:
            return None
        data += chunk
    return data


while True:
    header = read_exact(4)
    if header is None:
        break
    (samples,) = struct.unpack("<I", header)
    payload = read_exact(samples * 4)
    if payload is None:
        break
    response = {
        "language": "ru",
        "segments": [
            {
                "start": 0.0,
                "end": 0.5,
                "text": "__TEXT__",
                "confidence": 0.9,
                "words": [
                    {
                        "word": "__TEXT__",
                        "start": 0.0,
                        "end": 0.5,
                        "confidence": 0.9
                    }
                ]
            }
        ]
    }
    print(json.dumps(response), flush=True)
"#;

    fn write_dummy_python_backend_script(dir: &tempfile::TempDir) -> PathBuf {
        let script_path = dir.path().join("dummy_whisper.py");
        let script_source = PYTHON_BACKEND_TEMPLATE.replace("__TEXT__", PYTHON_TEST_TRANSCRIPT);

        stdfs::write(&script_path, script_source).expect("write dummy whisper backend script");
        script_path
    }

    fn python_backend_config() -> (WhisperConfig, tempfile::TempDir) {
        let dir = tempdir().expect("temp dir");
        let script_path = write_dummy_python_backend_script(&dir);

        let model_dir = dir.path().join("dummy_whisper_model");
        stdfs::create_dir_all(&model_dir).expect("create dummy whisper model dir");
        stdfs::write(model_dir.join("model.bin"), b"stub").expect("write dummy whisper model");

        let mut backend = WhisperPythonConfig::default();
        backend.script = script_path;

        let mut config = WhisperConfig::default();
        config.backend = WhisperBackendConfig::Python(backend);
        config.request_timeout_ms = 5_000;
        config.initialization_timeout_ms = 5_000;
        config.model.model_path = model_dir;

        (config, dir)
    }

    fn with_overridden_root<F: FnOnce()>(root: &Path, action: F) {
        let guard = env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var("KLARNET_ROOT").ok();
        std::env::set_var("KLARNET_ROOT", root);

        let result = catch_unwind(AssertUnwindSafe(action));

        if let Some(value) = previous {
            std::env::set_var("KLARNET_ROOT", value);
        } else {
            std::env::remove_var("KLARNET_ROOT");
        }

        drop(guard);
        result.unwrap();
    }

    #[test]
    fn resolves_target_release_model_when_project_stub_missing() {
        let project = tempdir().expect("project dir");
        let project_models = project.path().join("models").join("whisper-medium");
        stdfs::create_dir_all(&project_models).expect("create project models");

        let release_models = project
            .path()
            .join("target")
            .join("release")
            .join("models")
            .join("whisper-medium");
        stdfs::create_dir_all(&release_models).expect("create release models");
        stdfs::write(release_models.join("model.bin"), b"weights").expect("write release weights");

        with_overridden_root(project.path(), || {
            let resolved = resolve_whisper_model_path(Path::new("models/whisper-medium"))
                .expect("resolve model path");
            let expected = release_models
                .canonicalize()
                .expect("canonicalize release models");
            assert_eq!(resolved, expected);
        });
    }

    #[test]
    fn errors_when_multiple_weighted_models_detected() {
        let project = tempdir().expect("project dir");

        let project_models = project.path().join("models").join("whisper-medium");
        stdfs::create_dir_all(&project_models).expect("create project models");
        stdfs::write(project_models.join("model.bin"), b"weights").expect("write project weights");

        let release_models = project
            .path()
            .join("target")
            .join("release")
            .join("models")
            .join("whisper-medium");
        stdfs::create_dir_all(&release_models).expect("create release models");
        stdfs::write(release_models.join("model.bin"), b"weights").expect("write release weights");

        with_overridden_root(project.path(), || {
            let err = resolve_whisper_model_path(Path::new("models/whisper-medium"))
                .expect_err("expected duplicate error");

            match err {
                KlarnetError::Stt(message) => {
                    assert!(message.contains("Multiple Whisper models"));
                    assert!(message.contains("models/whisper-medium"));
                    assert!(message.contains("target/release"));
                }
                other => panic!("unexpected error: {other:?}"),
            }
        });
    }

    #[derive(Clone)]
    struct MockProcess {
        responses: Arc<Mutex<VecDeque<MockResult>>>,
        restarts: Arc<AtomicUsize>,
    }

    struct MockResult {
        delay: Option<Duration>,
        result: KlarnetResult<WhisperResponse>,
    }

    impl MockProcess {
        fn new(results: Vec<MockResult>) -> Self {
            Self {
                responses: Arc::new(Mutex::new(results.into_iter().collect())),
                restarts: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn success(text: &str) -> Self {
            let word = WhisperWord {
                word: text.to_string(),
                start: 0.0,
                end: 0.5,
                confidence: Some(0.9),
                probability: None,
            };

            let segment = WhisperSegment {
                start: 0.0,
                end: 0.5,
                text: text.to_string(),
                confidence: Some(0.9),
                words: vec![word],
            };

            Self::new(vec![MockResult {
                delay: None,
                result: Ok(WhisperResponse {
                    language: Some("ru".to_string()),
                    segments: vec![segment],
                }),
            }])
        }

        fn with_timeout_then_success() -> Self {
            let delayed = MockResult {
                delay: Some(Duration::from_millis(100)),
                result: Ok(WhisperResponse {
                    language: Some("ru".to_string()),
                    segments: vec![WhisperSegment {
                        start: 0.0,
                        end: 1.0,
                        text: "ignored".to_string(),
                        confidence: Some(0.5),
                        words: vec![],
                    }],
                }),
            };

            let success = MockResult {
                delay: None,
                result: Ok(WhisperResponse {
                    language: Some("ru".to_string()),
                    segments: vec![WhisperSegment {
                        start: 0.0,
                        end: 0.5,
                        text: "hello".to_string(),
                        confidence: Some(0.9),
                        words: vec![WhisperWord {
                            word: "hello".to_string(),
                            start: 0.0,
                            end: 0.5,
                            confidence: Some(0.9),
                            probability: None,
                        }],
                    }],
                }),
            };

            Self::new(vec![delayed, success])
        }

        fn restarts(&self) -> usize {
            self.restarts.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl WhisperBackend for MockProcess {
        async fn initialize(&mut self, _config: &WhisperConfig) -> KlarnetResult<()> {
            Ok(())
        }

        async fn transcribe(
            &mut self,
            _chunk: &AudioChunk,
            _pcm: &[f32],
            _config: &WhisperConfig,
        ) -> KlarnetResult<WhisperResponse> {
            let next = {
                let mut guard = self.responses.lock().await;
                guard.pop_front().unwrap()
            };

            if let Some(delay) = next.delay {
                sleep(delay).await;
            }

            next.result
        }

        async fn restart(&mut self, _config: &WhisperConfig) -> KlarnetResult<()> {
            self.restarts.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn shutdown(&mut self) {}
    }

    fn make_chunk() -> AudioChunk {
        let config = AudioConfig::default();
        let frame = AudioFrame {
            data: Arc::from(vec![0.0f32; config.buffer_size].into_boxed_slice()),
            timestamp: Utc::now(),
            duration: Duration::from_secs_f32(
                config.buffer_size as f32 / config.sample_rate as f32,
            ),
            sample_rate: config.sample_rate,
        };

        AudioChunk::new(vec![frame])
    }

    #[tokio::test]
    async fn python_backend_initializes_with_dummy_script() {
        let (config, _temp_dir) = python_backend_config();

        {
            let mut engine = WhisperEngine::new(config)
                .await
                .expect("python backend initializes");
            engine.shutdown();
        }
    }

    #[tokio::test]
    async fn python_backend_transcribes_with_dummy_script() {
        let (config, _temp_dir) = python_backend_config();

        {
            let mut engine = WhisperEngine::new(config).await.expect("engine init");

            let chunk = make_chunk();
            let transcript = engine
                .transcribe(chunk.clone())
                .await
                .expect("transcription success");

            assert_eq!(transcript.full_text, PYTHON_TEST_TRANSCRIPT);
            assert_eq!(transcript.language, "ru");
            assert_eq!(transcript.segments.len(), 1);

            engine.shutdown();
        }
    }

    #[tokio::test]
    async fn transcribe_with_mock_process() {
        let mut config = WhisperConfig::default();
        config.request_timeout_ms = 1_000;
        let backend: BackendHandle = Box::new(MockProcess::success("привет"));
        let mut engine = WhisperEngine::with_backend(config.clone(), backend)
            .await
            .expect("engine init");

        let chunk = make_chunk();
        let transcript = engine
            .transcribe(chunk.clone())
            .await
            .expect("transcription success");

        assert_eq!(transcript.full_text, "привет");
        assert_eq!(transcript.language, "ru");
        assert_eq!(transcript.segments.len(), 1);
        assert_eq!(transcript.segments[0].words.len(), 1);

        let metrics = engine.get_metrics();
        assert_eq!(metrics.total_processed, 1);
        assert_eq!(metrics.total_errors, 0);
        assert_eq!(metrics.total_retries, 0);
    }

    #[tokio::test]
    async fn timeout_retries_and_recovers() {
        let mut config = WhisperConfig::default();
        config.request_timeout_ms = 20;
        config.retry_attempts = 1;
        config.retry_backoff_ms = 1;

        let backend: BackendHandle = Box::new(MockProcess::with_timeout_then_success());
        let mut engine = WhisperEngine::with_backend(config.clone(), backend)
            .await
            .expect("engine init");

        let chunk = make_chunk();
        let transcript = engine
            .transcribe(chunk)
            .await
            .expect("transcription after retry");
        assert_eq!(transcript.full_text, "hello");

        let metrics = engine.get_metrics();
        assert_eq!(metrics.total_processed, 1);
        assert_eq!(metrics.total_timeouts, 1);
        assert_eq!(metrics.total_retries, 1);
        assert_eq!(metrics.total_restarts, 1);
    }

    #[tokio::test]
    async fn fails_after_retry_exhaustion() {
        let mut config = WhisperConfig::default();
        config.request_timeout_ms = 10;
        config.retry_attempts = 1;
        config.retry_backoff_ms = 1;

        let backend: BackendHandle = Box::new(MockProcess::new(vec![
            MockResult {
                delay: None,
                result: Err(KlarnetError::Stt("backend error".to_string())),
            },
            MockResult {
                delay: None,
                result: Err(KlarnetError::Stt("backend error".to_string())),
            },
        ]));

        let mut engine = WhisperEngine::with_backend(config.clone(), backend)
            .await
            .expect("engine init");

        let chunk = make_chunk();
        let err = engine.transcribe(chunk).await.expect_err("should fail");
        assert!(matches!(err, KlarnetError::Stt(_)));

        let metrics = engine.get_metrics();
        assert_eq!(metrics.total_processed, 0);
        assert_eq!(metrics.total_errors, 2);
        assert_eq!(metrics.total_retries, 1);
    }

    #[tokio::test]
    async fn rejects_invalid_language() {
        let mut config = WhisperConfig::default();
        config.language = "unsupported".to_string();
        let result = WhisperEngine::new(config).await;
        assert!(matches!(result, Err(KlarnetError::Config(_))));
    }
}
