#!/usr/bin/env python3
"""Simple stdin/stdout bridge for running faster-whisper as a subprocess."""

import argparse
import json
import logging
import struct
import sys
from typing import Any, Dict, List
import os

import numpy as np
from faster_whisper import WhisperModel

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a faster-whisper inference server")
    parser.add_argument("--model-path", required=True, help="Path to the Whisper model directory")
    parser.add_argument("--language", default="ru", help="Spoken language to force during decoding")
    parser.add_argument(
        "--compute-type",
        default="int8_float16",
        help="Compute type to pass to faster-whisper",
    )
    parser.add_argument("--device", default="cpu", help="Device on which to run the model")
    return parser.parse_args()


def read_exact(stream: Any, size: int) -> bytes:
    """Read exactly ``size`` bytes from ``stream`` or return an empty bytes object on EOF."""
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            return b""
        data.extend(chunk)
    return bytes(data)


def build_response(segments, info) -> Dict[str, Any]:
    response: Dict[str, Any] = {"segments": []}

    if hasattr(info, "language"):
        response["language"] = info.language
    if hasattr(info, "language_probability") and info.language_probability is not None:
        response["language_probability"] = float(info.language_probability)
    if hasattr(info, "duration") and info.duration is not None:
        response["duration"] = float(info.duration)
    if hasattr(info, "temperature") and info.temperature is not None:
        response["temperature"] = float(info.temperature)
    if hasattr(info, "avg_logprob") and info.avg_logprob is not None:
        response["avg_logprob"] = float(info.avg_logprob)
    if hasattr(info, "compression_ratio") and info.compression_ratio is not None:
        response["compression_ratio"] = float(info.compression_ratio)
    if hasattr(info, "no_speech_prob") and info.no_speech_prob is not None:
        response["no_speech_prob"] = float(info.no_speech_prob)

    for segment in segments:
        segment_payload: Dict[str, Any] = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "words": [],
        }

        if segment.words:
            words: List[Dict[str, Any]] = []
            for word in segment.words:
                probability = (
                    float(word.probability)
                    if word.probability is not None
                    else None
                )
                words.append(
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": probability,
                    }
                )
            segment_payload["words"] = words

        if hasattr(segment, "avg_logprob") and segment.avg_logprob is not None:
            segment_payload["avg_logprob"] = float(segment.avg_logprob)
        if hasattr(segment, "compression_ratio") and segment.compression_ratio is not None:
            segment_payload["compression_ratio"] = float(segment.compression_ratio)
        if hasattr(segment, "no_speech_prob") and segment.no_speech_prob is not None:
            segment_payload["no_speech_prob"] = float(segment.no_speech_prob)

        response["segments"].append(segment_payload)

    return response


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    # Load model
    model_path = args.model_path
    if model_path.startswith("\\\\?\\"):
        model_path = model_path[4:]
    model_path = os.path.normpath(model_path)

    model = WhisperModel(
        model_path,
        device=args.device,
        compute_type=args.compute_type,
    )

    LOGGER.info("Model loaded from %s", model_path)
    stdin_buffer = sys.stdin.buffer
    stdout_buffer = sys.stdout

    while True:
        # Читаем первые 4 байта - длину блока
        length_bytes = read_exact(stdin_buffer, 4)
        if not length_bytes:
            LOGGER.info("EOF reached – terminating")
            break

        # Распаковываем длину (число сэмплов)
        (sample_count,) = struct.unpack("<I", length_bytes)

        # Читаем PCM данные
        pcm_bytes = read_exact(stdin_buffer, sample_count * 4)
        if not pcm_bytes:
            LOGGER.warning("PCM payload missing – terminating")
            break

        pcm = np.frombuffer(pcm_bytes, dtype=np.float32)

        try:
            segments_iter, info = model.transcribe(
                pcm,
                language=args.language,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,  # Включить VAD фильтрацию
            )
            segments = list(segments_iter)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Error during transcription: %s", exc)
            print(json.dumps({"error": str(exc)}), file=sys.stderr, flush=True)
            continue

        # Формируем ответ
        response = build_response(segments, info)

        stdout_buffer.write(json.dumps(response) + "\n")
        stdout_buffer.flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.info("Interrupted – shutting down")
