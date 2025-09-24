import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "silero_tts.py"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "silero" / "v4_ru.pt"


pytestmark = pytest.mark.skipif(
    not SCRIPT_PATH.exists() or not MODEL_PATH.exists(),
    reason="Silero helper or model not available",
)


def _run_helper(extra_args=None):
    args = [
        sys.executable,
        "-u",
        str(SCRIPT_PATH),
        "--model",
        str(MODEL_PATH),
        "--speaker",
        "aidar",
        "--sample-rate",
        "48000",
        "--device",
        "cpu",
        "--target-rms",
        "0.15",
        "--log-level",
        "DEBUG",
    ]
    if extra_args:
        args.extend(extra_args)
    return subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _communicate(proc, payload, timeout=120):
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write(json.dumps(payload).encode("utf-8") + b"\n")
    proc.stdin.flush()

    start = time.time()
    line = b""
    while b"\n" not in line:
        chunk = proc.stdout.readline()
        if not chunk:
            break
        line = chunk
        if time.time() - start > timeout:
            raise TimeoutError("Timed out waiting for helper header")

    if not line:
        raise RuntimeError("Helper closed stdout unexpectedly")

    header = json.loads(line.decode("utf-8").strip())
    pcm = b""
    pcm_len = header.get("pcm_len", 0) if header.get("status") == "ok" else 0
    if pcm_len:
        while len(pcm) < pcm_len:
            chunk = proc.stdout.read(pcm_len - len(pcm))
            if not chunk:
                raise RuntimeError("Unexpected EOF while reading PCM payload")
            pcm += chunk
    return header, pcm


def test_silero_helper_synthesizes_audio():
    proc = _run_helper()
    try:
        request = {
            "id": "test-001",
            "text": "Привет, мир!",
            "speaker": "aidar",
            "sample_rate": 48000,
            "speed": 1.0,
        }
        header, pcm = _communicate(proc, request)

        assert header["status"] == "ok"
        assert header["id"] == request["id"]
        assert header["pcm_len"] == len(pcm)
        assert header["quality"] >= 0
        assert header["duration_ms"] > 0
        assert len(pcm) % 2 == 0

        samples = np.frombuffer(pcm, dtype="<i2")
        assert samples.size > 0
        assert np.max(np.abs(samples)) > 0
    finally:
        proc.terminate()
        proc.wait(timeout=60)


def test_silero_helper_reports_invalid_speaker():
    proc = _run_helper()
    try:
        request = {
            "id": "test-err",
            "text": "Ошибка",
            "speaker": "nonexistent",
            "sample_rate": 48000,
        }
        header, pcm = _communicate(proc, request)
        assert header["status"] == "error"
        assert "Speaker" in header["message"]
        assert pcm == b""
    finally:
        proc.terminate()
        proc.wait(timeout=60)