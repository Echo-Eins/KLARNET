// Python script for faster-whisper (scripts/whisper_server.py)
const WHISPER_PYTHON_SCRIPT: &str = r#"
#!/usr/bin/env python3
import sys
import json
import numpy as np
from faster_whisper import WhisperModel
import argparse
import logging
import struct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--language', default='ru')
    parser.add_argument('--compute-type', default='int8_float16')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # Load model
    model = WhisperModel(
        args.model_path,
        device=args.device,
        compute_type=args.compute_type
    )

    logger.info(f"Model loaded: {args.model_path}")

    while True:
        try:
            # Read length prefix
            length_bytes = sys.stdin.buffer.read(4)
            if not length_bytes:
                break

            length = struct.unpack('I', length_bytes)[0]

            # Read PCM data
            pcm_bytes = sys.stdin.buffer.read(length * 4)
            pcm = np.frombuffer(pcm_bytes, dtype=np.float32)

            # Transcribe
            segments, info = model.transcribe(
                pcm,
                language=args.language,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True
            )

            result = {
                'language': info.language,
                'segments': []
            }

            for segment in segments:
                seg_data = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text,
                    'words': []
                }

                if segment.words:
                    for word in segment.words:
                        seg_data['words'].append({
                            'word': word.word,
                            'start': word.start,
                            'end': word.end,
                            'probability': word.probability
                        })

                result['segments'].append(seg_data)

            # Send result
            sys.stdout.write(json.dumps(result) + '\n')
            sys.stdout.flush()

        except Exception as e:
            logger.error(f"Error: {e}")
            sys.stderr.write(f"Error: {e}\n")
            sys.stderr.flush()

if __name__ == '__main__':
    main()
"#;