from pathlib import Path

import numpy as np
import pytest

from persian_live_transcriber.audio import (
    AudioDevice,
    CombinedRecorder,
    audio_rms,
    find_blackhole_device,
    find_default_input_device,
    format_timestamp,
    list_input_devices,
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


def test_audio_rms() -> None:
    samples = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    assert audio_rms(samples) == 1.0


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


def test_list_input_devices_marks_blackhole_as_system_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDefault:
        device = [0, None]

    class FakeSoundDevice:
        default = FakeDefault()

        @staticmethod
        def query_devices() -> list[dict[str, object]]:
            return [
                {"name": "BlackHole 2ch", "max_input_channels": 2, "default_samplerate": 48_000},
                {
                    "name": "MacBook Pro Microphone",
                    "max_input_channels": 1,
                    "default_samplerate": 48_000,
                },
                {"name": "MacBook Pro Speakers", "max_input_channels": 0},
            ]

    monkeypatch.setattr("persian_live_transcriber.audio._import_sounddevice", lambda: FakeSoundDevice)

    devices = list_input_devices()

    assert [device.name for device in devices] == ["BlackHole 2ch", "MacBook Pro Microphone"]
    assert devices[0].is_system_audio is True
    assert devices[0].is_blackhole is True
    assert devices[1].is_system_audio is False
    assert find_blackhole_device(devices) == 0
    assert find_default_input_device(devices) == 1


def test_combined_recorder_rejects_microphone_as_system_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "persian_live_transcriber.audio.list_input_devices",
        lambda: [
            AudioDevice(
                id=2,
                name="MacBook Pro Microphone",
                max_input_channels=1,
                default_sample_rate=48_000,
                is_system_audio=False,
                is_blackhole=False,
                is_default_input=True,
            )
        ],
    )

    recorder = CombinedRecorder(source="system", mic_device=None, system_device=2)

    with pytest.raises(RuntimeError, match="not a BlackHole/system audio device"):
        recorder.start()
