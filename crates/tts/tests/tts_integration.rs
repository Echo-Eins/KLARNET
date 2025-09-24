use std::io::Write;
use std::path::PathBuf;

use tempfile::{NamedTempFile, TempDir};
use tts::silero::SileroTts;
use tts::{TtsBackend, TtsConfig, TtsEngine, TtsEngineType, TtsRuntimeConfig};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn silero_engine_produces_cached_pcm() {
    let mut script = NamedTempFile::new().unwrap();
    writeln!(
        script,
        "{}",
        r#"import sys, json
for line in sys.stdin:
    if not line.strip():
        continue
    req = json.loads(line)
    pcm = b"\x01\x00" * 8
    header = {"id": req["id"], "status": "ok", "pcm_len": len(pcm)}
    sys.stdout.write(json.dumps(header) + "\n")
    sys.stdout.flush()
    sys.stdout.buffer.write(pcm)
    sys.stdout.flush()
"#
    )
        .unwrap();
    script.flush().unwrap();

    let cache_dir = TempDir::new().unwrap();
    let mut config = TtsConfig::default();
    config.engine = TtsEngineType::Silero;
    config.runtime = TtsRuntimeConfig {
        python_path: PathBuf::from("python3"),
        silero_script: script.path().to_path_buf(),
        piper_binary: PathBuf::from("piper"),
        request_timeout_ms: 2_000,
    };
    config.cache.enabled = true;
    config.cache.directory = cache_dir.path().join("tts-cache");
    config.cache.max_entries = 4;
    config.monitoring.enabled = false;

    let engine = TtsEngine::new(config).await.unwrap();
    engine.speak("integration hello").await.unwrap();

    let cached = engine
        .cached_pcm_for_test("integration hello")
        .expect("entry in cache");
    assert!(!cached.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn piper_engine_reports_failure() {
    let mut binary = NamedTempFile::new().unwrap();
    writeln!(
        binary,
        "{}",
        r#"#!/usr/bin/env python3
import sys
sys.stderr.write("error\n")
sys.exit(3)
"#
    )
        .unwrap();
    binary.flush().unwrap();
    #[cfg(unix)]
    {
        use std::fs::Permissions;
        use std::os::unix::fs::PermissionsExt;
        binary
            .as_file_mut()
            .set_permissions(Permissions::from_mode(0o755))
            .unwrap();
    }

    let mut config = TtsConfig::default();
    config.engine = TtsEngineType::Piper;
    config.runtime = TtsRuntimeConfig {
        python_path: PathBuf::from("python3"),
        silero_script: PathBuf::from("scripts/silero_tts.py"),
        piper_binary: binary.path().to_path_buf(),
        request_timeout_ms: 1_000,
    };
    config.cache.enabled = false;
    config.monitoring.enabled = false;

    let engine = TtsEngine::new(config).await.unwrap();
    let err = engine.speak("should fail").await.unwrap_err();
    assert!(matches!(err, klarnet_core::KlarnetError::Action(_)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn direct_silero_backend_yields_pcm() {
    let mut script = NamedTempFile::new().unwrap();
    writeln!(
        script,
        "{}",
        r#"import sys, json
for line in sys.stdin:
    if not line.strip():
        continue
    req = json.loads(line)
    pcm = b"\x01\x00" * 6
    header = {"id": req["id"], "status": "ok", "pcm_len": len(pcm)}
    sys.stdout.write(json.dumps(header) + "\n")
    sys.stdout.flush()
    sys.stdout.buffer.write(pcm)
    sys.stdout.flush()
"#
    )
        .unwrap();
    script.flush().unwrap();

    let config = TtsConfig {
        runtime: TtsRuntimeConfig {
            python_path: PathBuf::from("python3"),
            silero_script: script.path().to_path_buf(),
            piper_binary: PathBuf::from("piper"),
            request_timeout_ms: 2_000,
        },
        ..TtsConfig::default()
    };
    let tts = SileroTts::new(config).await.unwrap();
    let pcm = tts.synthesize("integration direct").await.unwrap();
    assert!(!pcm.is_empty());
}