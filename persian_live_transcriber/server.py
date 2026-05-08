from __future__ import annotations

import asyncio
import importlib.util
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .asr import DEFAULT_MODEL, WhisperASR, mlx_whisper_available
from .audio import (
    DEFAULT_CHUNK_SECONDS,
    TARGET_SAMPLE_RATE,
    AudioSource,
    CombinedRecorder,
    audio_rms,
    find_blackhole_device,
    format_timestamp,
    list_input_devices,
    write_wav,
)
from .ollama_cleaner import DEFAULT_OLLAMA_MODEL, OllamaCleaner


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
SPEECH_RMS_THRESHOLD = 0.006
END_SILENCE_SECONDS = 1.0
MIN_SEGMENT_SECONDS = 0.9
CLEANUP_TIMEOUT_SECONDS = 20.0
COMMON_SILENCE_HALLUCINATIONS = {
    "موسیقی",
    "موسیقی در اینجا",
    "موسیقی.",
    "ممنون",
    "ممنون.",
}

app = FastAPI(title="Persian Live Transcriber", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@app.get("/api/status")
async def status() -> dict[str, Any]:
    devices = []
    device_error = None
    try:
        devices = list_input_devices()
    except Exception as exc:  # noqa: BLE001
        device_error = str(exc)

    ollama = OllamaCleaner().status()
    return {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "python": "3.11",
        "mlxWhisper": mlx_whisper_available(),
        "sounddevice": _module_available("sounddevice"),
        "ollama": {
            "available": ollama.available,
            "model": DEFAULT_OLLAMA_MODEL,
            "modelAvailable": ollama.model_available,
            "error": ollama.error,
        },
        "blackhole": {
            "available": find_blackhole_device(devices) is not None if devices else False,
            "installHint": "brew install --cask blackhole-2ch",
        },
        "audio": {
            "inputCount": len(devices),
            "error": device_error,
        },
        "asr": {
            "model": DEFAULT_MODEL,
            "language": "fa",
            "sampleRate": TARGET_SAMPLE_RATE,
            "chunkSeconds": DEFAULT_CHUNK_SECONDS,
        },
    }


@app.get("/api/devices")
async def devices() -> dict[str, Any]:
    try:
        items = [device.__dict__ for device in list_input_devices()]
        return {"devices": items, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"devices": [], "error": str(exc)}


async def _send(websocket: WebSocket, event: str, **payload: Any) -> None:
    try:
        await websocket.send_json({"event": event, **payload})
    except (RuntimeError, WebSocketDisconnect):
        return


async def _watch_stop(websocket: WebSocket, stop_event: asyncio.Event) -> None:
    try:
        while not stop_event.is_set():
            message = await websocket.receive_json()
            if message.get("action") == "stop":
                stop_event.set()
                return
    except WebSocketDisconnect:
        stop_event.set()


async def _cleanup_and_send(
    websocket: WebSocket,
    cleaner: OllamaCleaner,
    payload: dict[str, Any],
    raw_text: str,
    semaphore: asyncio.Semaphore,
) -> None:
    await _send(websocket, "cleaned", **{**payload, "text": raw_text}, cleanupPending=True)
    async with semaphore:
        try:
            cleaned = await asyncio.wait_for(
                asyncio.to_thread(cleaner.clean, raw_text, CLEANUP_TIMEOUT_SECONDS),
                timeout=CLEANUP_TIMEOUT_SECONDS + 5.0,
            )
            await _send(websocket, "cleaned", **{**payload, "text": cleaned})
        except Exception as exc:  # noqa: BLE001
            await _send(websocket, "cleaned", **{**payload, "text": raw_text}, cleanupFailed=True)
            await _send(
                websocket,
                "status",
                message=f"پاک‌سازی Ollama به‌موقع جواب نداد؛ متن خام نمایش داده شد. {exc}",
            )


@app.websocket("/ws/transcribe")
async def transcribe_ws(
    websocket: WebSocket,
    source: AudioSource = Query("mic"),
    mic_device: int | None = Query(default=None),
    system_device: int | None = Query(default=None),
    cleanup: bool = Query(default=True),
    chunk_seconds: float = Query(default=DEFAULT_CHUNK_SECONDS, ge=3.0, le=20.0),
) -> None:
    await websocket.accept()
    recorder = CombinedRecorder(source=source, mic_device=mic_device, system_device=system_device)
    asr = WhisperASR()
    cleaner = OllamaCleaner()
    segment_parts: list[np.ndarray] = []
    audio_elapsed = 0.0
    segment_start = 0.0
    silence_seconds = 0.0
    segment_index = 0
    stop_event = asyncio.Event()
    stop_task: asyncio.Task[None] | None = None
    cleanup_semaphore = asyncio.Semaphore(1)
    cleanup_tasks: set[asyncio.Task[None]] = set()
    last_waiting_status = 0.0

    try:
        recorder.start()
        stop_task = asyncio.create_task(_watch_stop(websocket, stop_event))
        await _send(
            websocket,
            "status",
            message="در حال ضبط؛ بعد از مکث کوتاه، بخش گفته‌شده ترنسکریپت می‌شود.",
            source=source,
            chunkSeconds=chunk_seconds,
            model=DEFAULT_MODEL,
        )

        while not stop_event.is_set():
            try:
                chunk = await asyncio.wait_for(recorder.read(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            chunk_duration = chunk.size / TARGET_SAMPLE_RATE
            chunk_start = audio_elapsed
            audio_elapsed += chunk_duration
            is_speech = audio_rms(chunk) >= SPEECH_RMS_THRESHOLD

            if not segment_parts and not is_speech:
                if audio_elapsed - last_waiting_status >= 4.0:
                    last_waiting_status = audio_elapsed
                    await _send(websocket, "status", message="منتظر گفتار واضح...")
                continue

            if not segment_parts:
                segment_start = chunk_start
                silence_seconds = 0.0

            segment_parts.append(chunk)
            if is_speech:
                silence_seconds = 0.0
            else:
                silence_seconds += chunk_duration

            segment_duration = audio_elapsed - segment_start
            should_finalize = (
                segment_duration >= chunk_seconds
                or (silence_seconds >= END_SILENCE_SECONDS and segment_duration >= MIN_SEGMENT_SECONDS)
            )
            if not should_finalize:
                continue

            buffered = np.concatenate(segment_parts)
            if silence_seconds > 0:
                trim_seconds = max(0.0, silence_seconds - 0.15)
                trim_samples = min(buffered.size - 1, int(trim_seconds * TARGET_SAMPLE_RATE))
                if trim_samples > 0:
                    buffered = buffered[:-trim_samples]

            start_time = segment_start
            end_time = max(segment_start, audio_elapsed - max(0.0, silence_seconds - 0.15))
            segment_parts = []
            silence_seconds = 0.0

            await _send(
                websocket,
                "partial",
                index=segment_index,
                start=max(0.0, start_time),
                end=end_time,
                startLabel=format_timestamp(max(0.0, start_time)),
                endLabel=format_timestamp(end_time),
                text="در حال ترنسکریپت بخش گفته‌شده...",
            )
            if not asr.is_loaded:
                await _send(
                    websocket,
                    "status",
                    message=(
                        "در حال بارگذاری مدل Whisper large-v3 از دیسک؛ این مرحله در اولین بخش چند "
                        "ثانیه طول می‌کشد."
                    ),
                )
            else:
                await _send(websocket, "status", message="در حال ترنسکریپت قطعه صوتی...")

            with tempfile.TemporaryDirectory(prefix="persian-transcript-") as tmpdir:
                wav_path = Path(tmpdir) / f"chunk-{segment_index}.wav"
                write_wav(wav_path, buffered, TARGET_SAMPLE_RATE)
                result = await asyncio.to_thread(asr.transcribe_file, wav_path)

            text = result.text.strip()
            if text in COMMON_SILENCE_HALLUCINATIONS:
                await _send(websocket, "status", message="صدای گفتار کافی تشخیص داده نشد.")
                segment_index += 1
                continue

            if text:
                payload = {
                    "index": segment_index,
                    "start": max(0.0, start_time),
                    "end": end_time,
                    "startLabel": format_timestamp(max(0.0, start_time)),
                    "endLabel": format_timestamp(end_time),
                    "text": text,
                    "language": result.language or "fa",
                }
                await _send(websocket, "final", **payload)
                await _send(websocket, "status", message="متن خام دریافت شد؛ ضبط ادامه دارد.")

                if cleanup:
                    cleanup_task = asyncio.create_task(
                        _cleanup_and_send(websocket, cleaner, payload, text, cleanup_semaphore)
                    )
                    cleanup_tasks.add(cleanup_task)
                    cleanup_task.add_done_callback(cleanup_tasks.discard)

            segment_index += 1

        await _send(websocket, "status", message="متوقف شد.")

    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001
        await _send(websocket, "error", message=str(exc))
    finally:
        if stop_task is not None:
            stop_task.cancel()
        for cleanup_task in list(cleanup_tasks):
            cleanup_task.cancel()
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        recorder.stop()
