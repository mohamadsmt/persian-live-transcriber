from __future__ import annotations

import asyncio
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

TARGET_SAMPLE_RATE = 16_000
DEFAULT_CHUNK_SECONDS = 10.0
SYSTEM_AUDIO_DEVICE_MARKERS = ("blackhole",)


AudioSource = Literal["mic", "system", "both"]


@dataclass(frozen=True)
class AudioDevice:
    id: int
    name: str
    max_input_channels: int
    default_sample_rate: int
    is_system_audio: bool
    is_blackhole: bool
    is_default_input: bool


def _import_sounddevice():
    import sounddevice as sd  # type: ignore

    return sd


def is_system_audio_device_name(name: str) -> bool:
    normalized = name.lower()
    return any(marker in normalized for marker in SYSTEM_AUDIO_DEVICE_MARKERS)


def list_input_devices() -> list[AudioDevice]:
    sd = _import_sounddevice()
    devices = sd.query_devices()
    default_input = sd.default.device[0] if sd.default.device else None
    result: list[AudioDevice] = []

    for idx, device in enumerate(devices):
        max_inputs = int(device.get("max_input_channels") or 0)
        if max_inputs <= 0:
            continue

        name = str(device.get("name") or f"Input {idx}")
        sample_rate = int(float(device.get("default_samplerate") or TARGET_SAMPLE_RATE))
        is_system_audio = is_system_audio_device_name(name)
        result.append(
            AudioDevice(
                id=idx,
                name=name,
                max_input_channels=max_inputs,
                default_sample_rate=sample_rate,
                is_system_audio=is_system_audio,
                is_blackhole="blackhole" in name.lower(),
                is_default_input=idx == default_input,
            )
        )

    return result


def find_default_input_device(devices: Iterable[AudioDevice] | None = None) -> int | None:
    candidates = [
        device
        for device in (list(devices) if devices is not None else list_input_devices())
        if not device.is_system_audio
    ]
    for device in candidates:
        if device.is_default_input:
            return device.id
    return candidates[0].id if candidates else None


def find_device_by_id(devices: Iterable[AudioDevice], device_id: int | None) -> AudioDevice | None:
    if device_id is None:
        return None
    for device in devices:
        if device.id == device_id:
            return device
    return None


def find_blackhole_device(devices: Iterable[AudioDevice] | None = None) -> int | None:
    candidates = list(devices) if devices is not None else list_input_devices()
    for device in candidates:
        if device.is_blackhole:
            return device.id
    return None


def to_mono(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(axis=1, dtype=np.float32)
    raise ValueError(f"Unsupported audio shape: {audio.shape}")


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    mono = to_mono(samples)
    if source_rate == target_rate or mono.size == 0:
        return mono.astype(np.float32, copy=False)

    duration = mono.size / float(source_rate)
    target_count = max(1, int(round(duration * target_rate)))
    source_axis = np.linspace(0.0, duration, num=mono.size, endpoint=False)
    target_axis = np.linspace(0.0, duration, num=target_count, endpoint=False)
    return np.interp(target_axis, source_axis, mono).astype(np.float32)


def mix_audio_tracks(tracks: Iterable[np.ndarray]) -> np.ndarray:
    prepared = [to_mono(track) for track in tracks if np.asarray(track).size]
    if not prepared:
        return np.zeros(0, dtype=np.float32)

    max_len = max(track.size for track in prepared)
    padded = []
    for track in prepared:
        if track.size < max_len:
            track = np.pad(track, (0, max_len - track.size))
        padded.append(track)

    mixed = np.mean(np.vstack(padded), axis=0, dtype=np.float32)
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def audio_rms(samples: np.ndarray) -> float:
    mono = to_mono(samples)
    if mono.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(mono, dtype=np.float32))))


def split_chunks(samples: np.ndarray, sample_rate: int, chunk_seconds: float) -> list[np.ndarray]:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    mono = to_mono(samples)
    size = max(1, int(round(sample_rate * chunk_seconds)))
    return [mono[start : start + size] for start in range(0, mono.size, size) if mono[start : start + size].size]


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    if millis == 1000:
        whole += 1
        millis = 0
    hours = whole // 3600
    minutes = (whole % 3600) // 60
    secs = whole % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    mono = np.clip(to_mono(samples), -1.0, 1.0)
    pcm = (mono * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


class DeviceRecorder:
    def __init__(self, device_id: int, block_seconds: float = 0.25):
        self.device_id = int(device_id)
        self.block_seconds = block_seconds
        self.sample_rate = TARGET_SAMPLE_RATE
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._stream = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        sd = _import_sounddevice()
        device_info = sd.query_devices(self.device_id, "input")
        self.sample_rate = int(float(device_info.get("default_samplerate") or TARGET_SAMPLE_RATE))
        blocksize = max(1, int(math.ceil(self.sample_rate * self.block_seconds)))
        self._loop = asyncio.get_running_loop()

        def callback(indata, frames, time, status):  # noqa: ANN001
            if status:
                # Status is not fatal; keep the stream alive and surface audio that arrived.
                pass
            chunk = resample_linear(np.copy(indata), self.sample_rate, TARGET_SAMPLE_RATE)
            if self._loop and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._queue.put_nowait, chunk)

        self._stream = sd.InputStream(
            device=self.device_id,
            channels=1,
            dtype="float32",
            samplerate=self.sample_rate,
            blocksize=blocksize,
            callback=callback,
        )
        self._stream.start()

    async def read(self) -> np.ndarray:
        return await self._queue.get()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None


class CombinedRecorder:
    def __init__(self, source: AudioSource, mic_device: int | None, system_device: int | None):
        self.source = source
        self.mic_device = mic_device
        self.system_device = system_device
        self.recorders: list[DeviceRecorder] = []

    def start(self) -> None:
        devices = list_input_devices()
        mic = self.mic_device if self.mic_device is not None else find_default_input_device(devices)
        system = self.system_device if self.system_device is not None else find_blackhole_device(devices)

        selected: list[int] = []
        if self.source in {"mic", "both"}:
            if mic is None:
                raise RuntimeError("No microphone input device is available.")
            mic_info = find_device_by_id(devices, mic)
            if mic_info is None:
                raise RuntimeError("Selected microphone input device is not available.")
            if mic_info.is_system_audio:
                raise RuntimeError("Selected microphone input device is a system audio device.")
            selected.append(mic)
        if self.source in {"system", "both"}:
            if system is None:
                raise RuntimeError("No system audio input device is available. Install/select BlackHole 2ch.")
            system_info = find_device_by_id(devices, system)
            if system_info is None:
                raise RuntimeError("Selected system audio input device is not available.")
            if not system_info.is_system_audio:
                raise RuntimeError("Selected system audio input device is not a BlackHole/system audio device.")
            selected.append(system)

        self.recorders = [DeviceRecorder(device_id) for device_id in selected]
        for recorder in self.recorders:
            recorder.start()

    async def read(self) -> np.ndarray:
        if not self.recorders:
            raise RuntimeError("Recorder has not been started.")
        chunks = await asyncio.gather(*(recorder.read() for recorder in self.recorders))
        return mix_audio_tracks(chunks)

    def stop(self) -> None:
        for recorder in self.recorders:
            recorder.stop()
        self.recorders = []
