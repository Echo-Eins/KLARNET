use std::path::PathBuf;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use klarnet_core::{
    AudioChunk, KlarnetError, KlarnetResult, Transcript, TranscriptSegment, WordInfo,
};
use serde::{Deserialize, Serialize};
use tokio::fs;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::time::{sleep, timeout};

const SUPPORTED_LANGUAGES: &[&str] = &[
    "af", "am", "ar", "as", "az", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy", "da", "de",
    "el", "en", "es", "et", "fa", "fi", "fr", "fy", "ga", "gd", "gl", "gu", "haw", "he", "hi",
    "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "la", "lb",
    "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn",
    "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq",
    "sr", "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "uz", "vi", "yi", "yo", "zh",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperModelConfig {
    pub model_path: PathBuf,
    #[serde(default = "WhisperModelConfig::default_device")]
    pub device: String,
    #[serde(default = "WhisperModelConfig::default_compute_type")]
    pub compute_type: String,
    #[serde(default)]
    pub cache_dir: Option<PathBuf>,
}

impl WhisperModelConfig {
    fn default_device() -> String {
        "cpu".to_string()
    }

    fn default_compute_type() -> String {
        "float16".to_string()
    }
}

impl Default for WhisperModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/whisper-medium"),
            device: Self::default_device(),
            compute_type: Self::default_compute_type(),
            cache_dir: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperPythonConfig {
    #[serde(default = "WhisperPythonConfig::default_executable")]
    pub executable: PathBuf,
    #[serde(default = "WhisperPythonConfig::default_script")]
    pub script: PathBuf,
    #[serde(default)]
    pub extra_args: Vec<String>,
    #[serde(default)]
    pub env: Vec<(String, String)>,
}

impl WhisperPythonConfig {
    fn default_executable() -> PathBuf {
        PathBuf::from("python3")
    }

    fn default_script() -> PathBuf {
        PathBuf::from("scripts/whisper_server.py")
    }
}

impl Default for WhisperPythonConfig {
    fn default() -> Self {
        Self {
            executable: Self::default_executable(),
            script: Self::default_script(),
            extra_args: Vec::new(),
            env: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WhisperBackendConfig {
    Python(WhisperPythonConfig),
    Native,
}

impl Default for WhisperBackendConfig {
    fn default() -> Self {
        Self::Python(WhisperPythonConfig::default())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperConfig {
    #[serde(default = "WhisperConfig::default_language")]
    pub language: String,
    #[serde(default)]
    pub model: WhisperModelConfig,
    #[serde(default)]
    pub backend: WhisperBackendConfig,
    #[serde(default = "WhisperConfig::default_request_timeout_ms")]
    pub request_timeout_ms: u64,
    #[serde(default = "WhisperConfig::default_initialization_timeout_ms")]
    pub initialization_timeout_ms: u64,
    #[serde(default = "WhisperConfig::default_retry_attempts")]
    pub retry_attempts: usize,
    #[serde(default = "WhisperConfig::default_retry_backoff_ms")]
    pub retry_backoff_ms: u64,
}

impl WhisperConfig {
    fn default_language() -> String {
        "ru".to_string()
    }

    fn default_request_timeout_ms() -> u64 {
        30_000
    }

    fn default_initialization_timeout_ms() -> u64 {
        120_000
    }

    fn default_retry_attempts() -> usize {
        2
    }

    fn default_retry_backoff_ms() -> u64 {
        500
    }

    pub fn request_timeout(&self) -> Duration {
        Duration::from_millis(self.request_timeout_ms)
    }

    pub fn initialization_timeout(&self) -> Duration {
        Duration::from_millis(self.initialization_timeout_ms)
    }

    pub fn retry_backoff(&self) -> Duration {
        Duration::from_millis(self.retry_backoff_ms)
    }

    pub fn validate(&self) -> KlarnetResult<()> {
        if self.language.trim().is_empty() {
            return Err(KlarnetError::Config(
                "Whisper language must not be empty".to_string(),
            ));
        }

        if !SUPPORTED_LANGUAGES
            .iter()
            .any(|lang| lang.eq_ignore_ascii_case(&self.language))
        {
            return Err(KlarnetError::Config(format!(
                "Unsupported Whisper language: {}",
                self.language
            )));
        }

        Ok(())
    }
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            language: Self::default_language(),
            model: WhisperModelConfig::default(),
            backend: WhisperBackendConfig::default(),
            request_timeout_ms: Self::default_request_timeout_ms(),
            initialization_timeout_ms: Self::default_initialization_timeout_ms(),
            retry_attempts: Self::default_retry_attempts(),
            retry_backoff_ms: Self::default_retry_backoff_ms(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct WhisperMetrics {
    pub total_processed: u64,
    pub total_errors: u64,
    pub total_timeouts: u64,
    pub total_retries: u64,
    pub total_restarts: u64,
    pub cumulative_processing_time: Duration,
    pub last_processing_time: Option<Duration>,
}

impl WhisperMetrics {
    pub fn average_processing_time(&self) -> Option<Duration> {
        if self.total_processed == 0 {
            return None;
        }

        Some(self.cumulative_processing_time / self.total_processed as u32)
    }
}

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
            WhisperBackendConfig::Python(python) => {
                Box::new(PythonWhisperProcess::new(python)) as BackendHandle
            }
            WhisperBackendConfig::Native => Box::new(NativeWhisper::default()) as BackendHandle,
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

        let mut command = Command::new(&self.config.executable);
        command
            .arg("-u")
            .arg(&self.config.script)
            .arg("--model-path")
            .arg(config.model.model_path.to_string_lossy().to_string())
            .arg("--language")
            .arg(&config.language)
            .arg("--compute-type")
            .arg(&config.model.compute_type)
            .arg("--device")
            .arg(&config.model.device)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

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

#[derive(Default)]
struct NativeWhisper;

#[async_trait]
impl WhisperBackend for NativeWhisper {
    async fn initialize(&mut self, _config: &WhisperConfig) -> KlarnetResult<()> {
        Err(KlarnetError::Stt(
            "Native Whisper backend is not implemented".to_string(),
        ))
    }

    async fn transcribe(
        &mut self,
        _chunk: &AudioChunk,
        _pcm: &[f32],
        _config: &WhisperConfig,
    ) -> KlarnetResult<WhisperResponse> {
        Err(KlarnetError::Stt(
            "Native Whisper backend is not implemented".to_string(),
        ))
    }

    fn shutdown(&mut self) {}
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;
    use chrono::Utc;
    use klarnet_core::{AudioConfig, AudioFrame};
    use tokio::sync::Mutex;

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
