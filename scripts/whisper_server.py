#!/usr/bin/env python3
"""Simple stdin/stdout bridge for running faster-whisper as a subprocess."""

import argparse
import json
import logging
import struct
import sys
from typing import Any, Dict, List

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


def build_response(segments) -> Dict[str, Any]:
    response: Dict[str, Any] = {"segments": []}
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
                words.append(
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability,
                    }
                )
            segment_payload["words"] = words

        response["segments"].append(segment_payload)

    return response


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    # Load model
    model = WhisperModel(
        args.model_path,
        device=args.device,
        compute_type=args.compute_type,
    )

    LOGGER.info("Model loaded from %s", args.model_path)
    stdin_buffer = sys.stdin.buffer
    stdout_buffer = sys.stdout

    while True:
        length_bytes = read_exact(stdin_buffer, 4)
                if not length_bytes:
                    LOGGER.info("EOF reached – terminating")
                    break

                (sample_count,) = struct.unpack("<I", length_bytes)
                pcm_bytes = read_exact(stdin_buffer, sample_count * 4)
                if not pcm_bytes:
                    LOGGER.warning("PCM payload missing – terminating")
                    break

                pcm = np.frombuffer(pcm_bytes, dtype=np.float32)

                try:
                    segments, info = model.transcribe(
                        pcm,
                        language=args.language,
                        beam_size=5,
                        word_timestamps=True,
                        vad_filter=True,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                        LOGGER.exception("Error during transcription: %s", exc)
                        print(json.dumps({"error": str(exc)}), file=sys.stderr, flush=True)
                        continue

                response = build_response(segments)
                response["language"] = info.language

                stdout_buffer.write(json.dumps(response) + "\n")
                stdout_buffer.flush()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.info("Interrupted – shutting down")