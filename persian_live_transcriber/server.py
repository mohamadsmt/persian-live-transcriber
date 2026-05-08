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
    find_blackhole_device,
    format_timestamp,
    list_input_devices,
    write_wav,
)
from .ollama_cleaner import DEFAULT_OLLAMA_MODEL, OllamaCleaner


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

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
    await websocket.send_json({"event": event, **payload})


async def _watch_stop(websocket: WebSocket, stop_event: asyncio.Event) -> None:
    try:
        while not stop_event.is_set():
            message = await websocket.receive_json()
            if message.get("action") == "stop":
                stop_event.set()
                return
    except WebSocketDisconnect:
        stop_event.set()


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
    audio_parts: list[np.ndarray] = []
    started_at = time.monotonic()
    segment_index = 0
    stop_event = asyncio.Event()
    stop_task: asyncio.Task[None] | None = None

    try:
        recorder.start()
        stop_task = asyncio.create_task(_watch_stop(websocket, stop_event))
        await _send(
            websocket,
            "status",
            message="recording",
            source=source,
            chunkSeconds=chunk_seconds,
            model=DEFAULT_MODEL,
        )

        while not stop_event.is_set():
            try:
                chunk = await asyncio.wait_for(recorder.read(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            audio_parts.append(chunk)
            buffered = np.concatenate(audio_parts) if audio_parts else np.zeros(0, dtype=np.float32)
            if buffered.size < int(TARGET_SAMPLE_RATE * chunk_seconds):
                continue

            audio_parts = []
            start_time = time.monotonic() - started_at - (buffered.size / TARGET_SAMPLE_RATE)
            end_time = time.monotonic() - started_at
            await _send(
                websocket,
                "partial",
                index=segment_index,
                start=max(0.0, start_time),
                end=end_time,
                text="در حال ترنسکریپت...",
            )

            with tempfile.TemporaryDirectory(prefix="persian-transcript-") as tmpdir:
                wav_path = Path(tmpdir) / f"chunk-{segment_index}.wav"
                write_wav(wav_path, buffered, TARGET_SAMPLE_RATE)
                result = await asyncio.to_thread(asr.transcribe_file, wav_path)

            text = result.text.strip()
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

                if cleanup:
                    try:
                        cleaned = await asyncio.to_thread(cleaner.clean, text)
                        await _send(websocket, "cleaned", **payload, text=cleaned)
                    except Exception as exc:  # noqa: BLE001
                        await _send(websocket, "error", message=f"Ollama cleanup failed: {exc}")

            segment_index += 1

        await _send(websocket, "status", message="stopped")

    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001
        await _send(websocket, "error", message=str(exc))
    finally:
        if stop_task is not None:
            stop_task.cancel()
        recorder.stop()
