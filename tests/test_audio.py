from pathlib import Path

import numpy as np

from persian_live_transcriber.audio import (
    format_timestamp,
    mix_audio_tracks,
    resample_linear,
    split_chunks,
    write_wav,
)


def test_resample_linear_to_16k() -> None:
    samples = np.ones(48_000, dtype=np.float32)
    result = resample_linear(samples, source_rate=48_000, target_rate=16_000)
    assert result.shape == (16_000,)
    assert np.allclose(result, 1.0)


def test_mix_audio_tracks_pads_and_averages() -> None:
    first = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    second = np.array([-1.0], dtype=np.float32)
    result = mix_audio_tracks([first, second])
    assert np.allclose(result, np.array([0.0, 0.5, 0.5], dtype=np.float32))


def test_split_chunks_uses_sample_rate() -> None:
    samples = np.arange(10, dtype=np.float32)
    chunks = split_chunks(samples, sample_rate=10, chunk_seconds=0.3)
    assert [chunk.tolist() for chunk in chunks] == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def test_format_timestamp_srt_shape() -> None:
    assert format_timestamp(3661.234) == "01:01:01,234"


def test_write_wav(tmp_path: Path) -> None:
    path = tmp_path / "sample.wav"
    write_wav(path, np.zeros(160, dtype=np.float32), 16_000)
    assert path.stat().st_size > 44

