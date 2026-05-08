from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from persian_live_transcriber.asr import DEFAULT_MODEL, WhisperASR
from persian_live_transcriber.audio import TARGET_SAMPLE_RATE, write_wav


def main() -> None:
    audio_path = Path("/private/tmp/persian-transcriber-preload.wav")
    write_wav(audio_path, np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32), TARGET_SAMPLE_RATE)

    started = time.monotonic()
    print(f"Preloading Whisper {DEFAULT_MODEL}...")
    result = WhisperASR().transcribe_file(audio_path)
    elapsed = time.monotonic() - started
    print(f"Model ready in {elapsed:.2f}s. Probe language={result.language!r}, text={result.text!r}")


if __name__ == "__main__":
    main()
