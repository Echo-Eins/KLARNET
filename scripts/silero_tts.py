#!/usr/bin/env python3
"""Silero TTS helper process.

This script exposes the Silero text-to-speech model over a stdio protocol that
matches the expectations of the Rust backend implemented in
``crates/tts/src/silero.rs``.  Requests are accepted as JSON objects written to
stdin, one object per line.  Each request must at minimum include an ``id`` and
``text`` field; the remaining parameters are optional and default to the values
provided on the command line when the helper is started.

Responses are written as a JSON header followed by raw PCM16LE audio bytes:

1. ``{"id": ..., "status": "ok", "pcm_len": N, ...}"``, terminated by a
   newline character.
2. Exactly ``N`` bytes containing the synthesized mono audio stream encoded as
   16-bit little-endian samples at the requested sample rate.

When synthesis fails the helper emits a header with ``status`` set to
``"error"`` and does not write any PCM payload.
"""

from __future__ import annotations

import argparse
import json
import logging
import select
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.package import PackageImporter

try:
    import torchaudio.functional as F
except Exception as exc:  # pragma: no cover - torchaudio is a hard dependency
    raise RuntimeError(
        "torchaudio is required for Silero TTS runtime. Please ensure it is "
        "installed alongside torch."
    ) from exc


LOGGER = logging.getLogger("silero_tts")


@dataclass(frozen=True)
class RuntimeArgs:
    """CLI arguments controlling the helper process."""

    model_path: Path
    speaker: str
    sample_rate: int
    speed: float
    device: str
    target_rms: Optional[float]
    allow_accent: bool
    allow_yo: bool
    preload: bool
    channels: int


class SignalHandler:
    """Small helper to coordinate graceful shutdown on POSIX signals."""

    def __init__(self) -> None:
        self._stop_event = threading.Event()

    def install(self) -> None:
        signal.signal(signal.SIGINT, self._handle)  # type: ignore[arg-type]
        signal.signal(signal.SIGTERM, self._handle)  # type: ignore[arg-type]

    def _handle(self, signum, _frame) -> None:  # pragma: no cover - signal
        LOGGER.info("Received signal %s, stopping helper", signum)
        self._stop_event.set()

    def should_stop(self) -> bool:
        return self._stop_event.is_set()


class ModelCache:
    """Load Silero models lazily and keep them resident for reuse."""

    def __init__(self) -> None:
        self._models: Dict[Tuple[Path, str], torch.nn.Module] = {}
        self._lock = threading.Lock()

    def get(self, model_path: Path, device: torch.device) -> torch.nn.Module:
        key = (model_path.resolve(), str(device))
        with self._lock:
            if key in self._models:
                return self._models[key]

            LOGGER.info("Loading Silero model from %s on %s", model_path, device)
            importer = PackageImporter(str(model_path))
            model = importer.load_pickle("tts_models", "model")
            if hasattr(model, "to"):
                model.to(device)
            try:
                speakers = list(list_speakers(model))
                if speakers:
                    preview = ", ".join(speakers[:10])
                    LOGGER.info(
                        "Model exposes %d speakers; sample: %s%s",
                        len(speakers),
                        preview,
                        "..." if len(speakers) > 10 else "",
                    )
            except Exception:  # noqa: BLE001
                LOGGER.debug("Loaded model does not expose speakers list", exc_info=True)
            self._models[key] = model
            return model


MODEL_CACHE = ModelCache()


def parse_args(argv: Optional[Iterable[str]] = None) -> RuntimeArgs:
    parser = argparse.ArgumentParser(description="Silero TTS stdio helper")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the Silero .pt model archive (e.g. v4_ru.pt)",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="aidar",
        help="Default speaker to use when a request does not override it",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48_000,
        help="Output sample rate in Hz",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Default synthesis speed multiplier",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to run inference on (e.g. cpu, cuda:0)",
    )
    parser.add_argument(
        "--target-rms",
        type=float,
        default=0.2,
        help=(
            "Target root-mean-square signal level for volume normalization. "
            "Set to 0 to disable normalization."
        ),
    )
    parser.add_argument(
        "--no-accent",
        dest="allow_accent",
        action="store_false",
        help="Disable accent restoration during synthesis",
    )
    parser.add_argument(
        "--no-yo",
        dest="allow_yo",
        action="store_false",
        help="Disable automatic Ё handling",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging level",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load the Silero model immediately rather than on first request",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of output channels (1 for mono, 2 for duplicated stereo)",
    )

    parser.set_defaults(allow_accent=True, allow_yo=True)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    model_path: Path = args.model
    if not model_path.is_file():
        parser.error(f"Model file not found: {model_path}")

    if args.channels not in (1, 2):
        parser.error("Only mono or stereo output is supported")

    target_rms = args.target_rms if args.target_rms > 0 else None

    runtime_args = RuntimeArgs(
        model_path=model_path,
        speaker=args.speaker,
        sample_rate=args.sample_rate,
        speed=args.speed,
        device=args.device,
        target_rms=target_rms,
        allow_accent=args.allow_accent,
        allow_yo=args.allow_yo,
        preload=args.preload,
        channels=args.channels,
    )

    LOGGER.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    LOGGER.debug("Parsed runtime arguments: %s", runtime_args)

    return runtime_args


def resolve_device(device: str) -> torch.device:
    try:
        torch_device = torch.device(device)
    except RuntimeError as exc:
        raise RuntimeError(f"Invalid torch device requested: {device}") from exc

    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available")

    return torch_device


def list_speakers(model: torch.nn.Module) -> Iterable[str]:
    if hasattr(model, "speakers"):
        return getattr(model, "speakers")
    raise AttributeError("Loaded Silero model does not expose speakers list")


def normalize_audio(samples: np.ndarray, target_rms: Optional[float]) -> Tuple[np.ndarray, float]:
    if samples.size == 0:
        return samples, 0.0

    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    if target_rms is None or rms == 0.0:
        return samples, rms

    gain = target_rms / rms
    normalized = samples * gain
    normalized = np.clip(normalized, -1.0, 1.0)
    new_rms = float(np.sqrt(np.mean(np.square(normalized), dtype=np.float64)))
    return normalized, new_rms


def duplicate_channels(samples: np.ndarray, channels: int) -> np.ndarray:
    if channels == 1:
        return samples
    return np.repeat(samples[:, None], channels, axis=1).reshape(-1)


def to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    int16 = np.round(clipped * np.iinfo(np.int16).max).astype("<i2")
    return int16.tobytes()


def resample_if_needed(
    samples: np.ndarray, input_rate: int, output_rate: int
) -> np.ndarray:
    if input_rate == output_rate:
        return samples

    waveform = torch.from_numpy(samples.astype(np.float32)).unsqueeze(0)
    resampled = F.resample(waveform, input_rate, output_rate)
    return resampled.squeeze(0).cpu().numpy()


def apply_speed(samples: np.ndarray, sample_rate: int, speed: float) -> Tuple[np.ndarray, int]:
    if np.isclose(speed, 1.0):
        return samples, sample_rate

    target_rate = max(1, int(round(sample_rate / speed)))
    waveform = torch.from_numpy(samples.astype(np.float32)).unsqueeze(0)
    adjusted = F.resample(waveform, sample_rate, target_rate)
    return adjusted.squeeze(0).cpu().numpy(), target_rate


def synthesize(
    model: torch.nn.Module,
    request: Dict[str, object],
    runtime: RuntimeArgs,
    device: torch.device,
) -> Tuple[bytes, float, float]:
    text = str(request.get("text", ""))
    if not text.strip():
        raise ValueError("Text payload is empty")

    speaker = str(request.get("speaker", runtime.speaker))
    available_speakers = list_speakers(model)
    if speaker not in available_speakers:
        raise ValueError(
            f"Speaker '{speaker}' is not available. Supported speakers: "
            f"{', '.join(available_speakers)}"
        )

    speed = float(request.get("speed", runtime.speed))
    if speed <= 0:
        raise ValueError("Speed multiplier must be positive")

    output_rate = int(request.get("sample_rate", runtime.sample_rate))
    if output_rate <= 0:
        raise ValueError("Sample rate must be positive")

    start = time.perf_counter()
    LOGGER.debug(
        "Starting synthesis: id=%s speaker=%s rate=%d speed=%.2f",  # noqa: G003
        request.get("id"),
        speaker,
        output_rate,
        speed,
    )

    native_rate = int(getattr(model, "sample_rate", runtime.sample_rate))
    audio = model.apply_tts(
        text=text,
        speaker=speaker,
        sample_rate=native_rate,
        put_accent=runtime.allow_accent,
        put_yo=runtime.allow_yo,
    )

    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().numpy()
    else:
        audio_np = np.asarray(audio, dtype=np.float32)

    if audio_np.ndim != 1:
        raise RuntimeError(
            "Silero returned unexpected audio shape: " f"{audio_np.shape}"
        )

    audio_np, native_rate = apply_speed(audio_np, native_rate, speed)
    audio_np = resample_if_needed(audio_np, native_rate, output_rate)

    normalized, rms = normalize_audio(audio_np, runtime.target_rms)
    normalized = duplicate_channels(normalized, runtime.channels)
    pcm = to_pcm16(normalized)
    elapsed = time.perf_counter() - start

    LOGGER.info(
        "Synthesis completed: id=%s duration_ms=%0.1f rms=%0.4f peak=%0.4f latency_ms=%0.1f",  # noqa: G003
        request.get("id"),
        len(normalized) / output_rate / runtime.channels * 1000.0,
        rms,
        float(np.max(np.abs(normalized))) if normalized.size else 0.0,
        elapsed * 1000.0,
    )

    quality_score = min(1.0, rms / 0.3) if rms > 0 else 0.0
    duration_ms = int((len(normalized) / runtime.channels) / output_rate * 1000)

    return pcm, quality_score, duration_ms


def main(argv: Optional[Iterable[str]] = None) -> int:
    runtime_args = parse_args(argv)
    signal_handler = SignalHandler()
    signal_handler.install()

    try:
        device = resolve_device(runtime_args.device)
    except RuntimeError as exc:
        LOGGER.error(str(exc))
        return 2

    if runtime_args.preload:
        MODEL_CACHE.get(runtime_args.model_path, device)

    LOGGER.info(
        "Silero helper ready: model=%s device=%s default_speaker=%s",
        runtime_args.model_path,
        device,
        runtime_args.speaker,
    )

    stdin_buffer = sys.stdin.buffer

    while not signal_handler.should_stop():
        if not stdin_buffer.closed:
            ready, _, _ = select.select([stdin_buffer], [], [], 0.1)
        else:
            ready = []

        if not ready:
            continue

        try:
            line_bytes = stdin_buffer.readline()
        except KeyboardInterrupt:  # pragma: no cover - handled by signal
            break

        if line_bytes == b"":
            LOGGER.info("EOF on stdin; shutting down")
            break

        stripped = line_bytes.decode("utf-8", errors="ignore").strip()
        if not stripped:
            continue

        try:
            request = json.loads(stripped)
        except json.JSONDecodeError as exc:
            LOGGER.warning("Failed to decode request: %s", exc)
            error_header = {
                "id": None,
                "status": "error",
                "message": f"Invalid JSON payload: {exc}",
            }
            sys.stdout.write(json.dumps(error_header) + "\n")
            sys.stdout.flush()
            continue

        if signal_handler.should_stop():
            break

        request_id = request.get("id")
        try:
            model = MODEL_CACHE.get(runtime_args.model_path, device)
            pcm, quality, duration_ms = synthesize(
                model=model,
                request=request,
                runtime=runtime_args,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Synthesis failed for request %s", request_id)
            header = {
                "id": request_id,
                "status": "error",
                "message": str(exc),
            }
            sys.stdout.write(json.dumps(header) + "\n")
            sys.stdout.flush()
            continue

        header = {
            "id": request_id,
            "status": "ok",
            "pcm_len": len(pcm),
            "quality": quality,
            "duration_ms": duration_ms,
        }
        sys.stdout.write(json.dumps(header) + "\n")
        sys.stdout.flush()
        sys.stdout.buffer.write(pcm)
        sys.stdout.flush()

    LOGGER.info("Silero helper stopped")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())