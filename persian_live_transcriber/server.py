from __future__ import annotations

import asyncio
import importlib.util
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .asr import DEFAULT_MODEL, TranscriptResult, WhisperASR, mlx_whisper_available
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
SUMMARY_TIMEOUT_SECONDS = 120.0
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
FFMPEG_TIMEOUT_SECONDS = 600.0
WHISPER_SEGMENT_FRAMES_PER_SECOND = 100.0
COMMON_SILENCE_HALLUCINATIONS = {
    "موسیقی",
    "موسیقی در اینجا",
    "موسیقی.",
    "ممنون",
    "ممنون.",
}

app = FastAPI(title="Persian Live Transcriber", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SummaryRequest(BaseModel):
    text: str = Field(default="", max_length=200_000)


class SummaryResponse(BaseModel):
    summary: str
    model: str


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


@app.post("/api/summarize", response_model=SummaryResponse)
async def summarize_session(request: SummaryRequest) -> SummaryResponse:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="متنی برای خلاصه‌سازی وجود ندارد.")

    cleaner = OllamaCleaner()
    try:
        summary = await asyncio.wait_for(
            asyncio.to_thread(cleaner.summarize, text, SUMMARY_TIMEOUT_SECONDS),
            timeout=SUMMARY_TIMEOUT_SECONDS + 5.0,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"خلاصه‌سازی Ollama انجام نشد: {exc}",
        ) from exc

    if not summary:
        raise HTTPException(status_code=503, detail="خلاصه‌سازی Ollama خروجی خالی برگرداند.")

    return SummaryResponse(summary=summary, model=DEFAULT_OLLAMA_MODEL)


def _safe_filename(filename: str | None) -> str | None:
    if not filename:
        return None
    decoded = unquote(filename)
    cleaned = Path(decoded).name.strip()
    return cleaned or None


async def _save_uploaded_audio(request: Request, destination: Path) -> int:
    total = 0
    with destination.open("wb") as file:
        async for chunk in request.stream():
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="فایل صوتی بزرگ‌تر از حد مجاز است.")
            file.write(chunk)

    if total <= 0:
        raise HTTPException(status_code=400, detail="فایل صوتی خالی است.")
    return total


def _convert_audio_to_wav(input_path: Path, output_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(status_code=503, detail="ffmpeg روی سیستم پیدا نشد.")

    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-f",
        "wav",
        str(output_path),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="تبدیل فایل صوتی بیش از حد طول کشید.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "فرمت فایل صوتی قابل تبدیل نیست.").strip()
        raise HTTPException(
            status_code=422,
            detail=f"تبدیل فایل صوتی با ffmpeg انجام نشد: {detail}",
        ) from exc


def _wav_duration_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            if frame_rate <= 0:
                return 0.0
            return wav_file.getnframes() / float(frame_rate)
    except wave.Error:
        return 0.0


def _coerce_seconds(value: Any, fallback: float) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return max(0.0, fallback)


def _segment_value(segment: Any, key: str) -> Any:
    if isinstance(segment, dict):
        return segment.get(key)
    if isinstance(segment, (list, tuple)):
        positions = {"start": 0, "end": 1, "text": 2}
        index = positions.get(key)
        if index is not None and len(segment) > index:
            return segment[index]
    return None


def _coerce_segment_seconds(segment: Any, key: str, fallback: float) -> float:
    value = _segment_value(segment, key)
    if value is None:
        return max(0.0, fallback)
    seconds = _coerce_seconds(value, fallback)
    if isinstance(segment, (list, tuple)):
        return seconds / WHISPER_SEGMENT_FRAMES_PER_SECOND
    return seconds


def _segment_payload(
    index: int,
    start: float,
    end: float,
    text: str,
    language: str,
) -> dict[str, Any]:
    end = max(start, end)
    return {
        "index": index,
        "start": start,
        "end": end,
        "startLabel": format_timestamp(start),
        "endLabel": format_timestamp(end),
        "text": text,
        "language": language,
    }


def _transcript_segments(result: TranscriptResult, duration: float) -> list[dict[str, Any]]:
    language = result.language or "fa"
    normalized: list[dict[str, Any]] = []
    last_end = 0.0

    for segment in result.segments or []:
        text = str(_segment_value(segment, "text") or "").strip()
        if not text or text in COMMON_SILENCE_HALLUCINATIONS:
            continue
        start = _coerce_segment_seconds(segment, "start", last_end)
        end = _coerce_segment_seconds(segment, "end", max(start, last_end))
        payload = _segment_payload(len(normalized), start, end, text, language)
        normalized.append(payload)
        last_end = payload["end"]

    if normalized:
        return normalized

    text = result.text.strip()
    if not text or text in COMMON_SILENCE_HALLUCINATIONS:
        return []
    return [_segment_payload(0, 0.0, max(0.0, duration), text, language)]


async def _clean_file_segments(
    segments: list[dict[str, Any]],
    cleaner: OllamaCleaner,
) -> list[dict[str, Any]]:
    cleaned_segments = []
    for segment in segments:
        text = str(segment.get("text") or "")
        try:
            cleaned = await asyncio.wait_for(
                asyncio.to_thread(cleaner.clean, text, CLEANUP_TIMEOUT_SECONDS),
                timeout=CLEANUP_TIMEOUT_SECONDS + 5.0,
            )
            cleaned_segments.append({**segment, "text": cleaned})
        except Exception:  # noqa: BLE001
            cleaned_segments.append({**segment, "cleanupFailed": True})
    return cleaned_segments


@app.post("/api/transcribe-file")
async def transcribe_file_upload(
    request: Request,
    cleanup: bool = Query(default=True),
    filename: str | None = Query(default=None),
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="persian-transcript-upload-") as tmpdir:
        upload_path = Path(tmpdir) / "upload.audio"
        wav_path = Path(tmpdir) / "upload.wav"
        byte_count = await _save_uploaded_audio(request, upload_path)
        await asyncio.to_thread(_convert_audio_to_wav, upload_path, wav_path)

        asr = WhisperASR()
        result = await asyncio.to_thread(asr.transcribe_file, wav_path)
        raw_segments = _transcript_segments(result, _wav_duration_seconds(wav_path))
        clean_segments = []
        if cleanup and raw_segments:
            clean_segments = await _clean_file_segments(raw_segments, OllamaCleaner())

    return {
        "filename": _safe_filename(filename),
        "byteCount": byte_count,
        "model": DEFAULT_MODEL,
        "language": result.language or "fa",
        "rawSegments": raw_segments,
        "cleanSegments": clean_segments,
    }


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
