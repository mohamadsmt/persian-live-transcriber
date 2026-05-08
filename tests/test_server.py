import asyncio
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from persian_live_transcriber.asr import TranscriptResult, WhisperASR
from persian_live_transcriber.audio import TARGET_SAMPLE_RATE, write_wav
from persian_live_transcriber.server import _cleanup_and_send, _transcript_segments, app


def _write_silent_wav(path: Path) -> None:
    write_wav(path, np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32), TARGET_SAMPLE_RATE)


def test_status_endpoint_shape() -> None:
    client = TestClient(app)
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["asr"]["model"] == "large-v3"
    assert data["asr"]["language"] == "fa"
    assert "ollama" in data


def test_index_serves_ui() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "ترنسکریپت زنده فارسی" in response.text


def test_summarize_endpoint_rejects_empty_text() -> None:
    client = TestClient(app)
    response = client.post("/api/summarize", json={"text": "   "})
    assert response.status_code == 400


def test_summarize_endpoint_returns_ollama_summary(monkeypatch) -> None:
    class FakeCleaner:
        def summarize(self, text: str, timeout: float) -> str:
            assert text == "متن کامل جلسه"
            assert timeout > 0
            return "## خلاصه کلی\nخلاصه مفصل جلسه"

    monkeypatch.setattr("persian_live_transcriber.server.OllamaCleaner", lambda: FakeCleaner())

    client = TestClient(app)
    response = client.post("/api/summarize", json={"text": "متن کامل جلسه"})

    assert response.status_code == 200
    assert response.json()["summary"] == "## خلاصه کلی\nخلاصه مفصل جلسه"
    assert response.json()["model"] == "gpt-oss:20b"


def test_summarize_endpoint_reports_ollama_error(monkeypatch) -> None:
    class FakeCleaner:
        def summarize(self, text: str, timeout: float) -> str:
            raise RuntimeError("ollama down")

    monkeypatch.setattr("persian_live_transcriber.server.OllamaCleaner", lambda: FakeCleaner())

    client = TestClient(app)
    response = client.post("/api/summarize", json={"text": "متن کامل جلسه"})

    assert response.status_code == 503
    assert "خلاصه‌سازی Ollama انجام نشد" in response.json()["detail"]


def test_transcribe_file_rejects_empty_upload() -> None:
    client = TestClient(app)
    response = client.post("/api/transcribe-file", content=b"")
    assert response.status_code == 400
    assert "خالی" in response.json()["detail"]


def test_transcribe_file_rejects_oversized_upload(monkeypatch) -> None:
    monkeypatch.setattr("persian_live_transcriber.server.MAX_UPLOAD_BYTES", 3)

    client = TestClient(app)
    response = client.post("/api/transcribe-file", content=b"audio")

    assert response.status_code == 413


def test_transcribe_file_returns_raw_segments(monkeypatch) -> None:
    class FakeASR:
        def transcribe_file(self, audio_path: Path) -> TranscriptResult:
            assert audio_path.name == "upload.wav"
            return TranscriptResult(
                text="سلام دنیا",
                language="fa",
                segments=[
                    {"start": 0.25, "end": 1.5, "text": "سلام"},
                    {"start": 1.5, "end": 2.0, "text": "دنیا"},
                ],
            )

    def fake_convert(input_path: Path, output_path: Path) -> None:
        assert input_path.read_bytes() == b"audio"
        _write_silent_wav(output_path)

    monkeypatch.setattr("persian_live_transcriber.server._convert_audio_to_wav", fake_convert)
    monkeypatch.setattr("persian_live_transcriber.server.WhisperASR", FakeASR)

    client = TestClient(app)
    response = client.post(
        "/api/transcribe-file?cleanup=false&filename=sample.mp3",
        content=b"audio",
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "sample.mp3"
    assert data["cleanSegments"] == []
    assert data["rawSegments"][0]["text"] == "سلام"
    assert data["rawSegments"][0]["startLabel"] == "00:00:00,250"
    assert data["rawSegments"][1]["endLabel"] == "00:00:02,000"


def test_transcribe_file_cleans_segments(monkeypatch) -> None:
    class FakeASR:
        def transcribe_file(self, audio_path: Path) -> TranscriptResult:
            return TranscriptResult(
                text="متن خام",
                language="fa",
                segments=[{"start": 0.0, "end": 1.0, "text": "متن خام"}],
            )

    class FakeCleaner:
        def clean(self, text: str, timeout: float) -> str:
            assert timeout > 0
            return f"{text}!"

    monkeypatch.setattr(
        "persian_live_transcriber.server._convert_audio_to_wav",
        lambda input_path, output_path: _write_silent_wav(output_path),
    )
    monkeypatch.setattr("persian_live_transcriber.server.WhisperASR", FakeASR)
    monkeypatch.setattr("persian_live_transcriber.server.OllamaCleaner", lambda: FakeCleaner())

    client = TestClient(app)
    response = client.post("/api/transcribe-file?cleanup=true", content=b"audio")

    assert response.status_code == 200
    assert response.json()["cleanSegments"][0]["text"] == "متن خام!"


def test_transcribe_file_keeps_raw_text_when_cleanup_fails(monkeypatch) -> None:
    class FakeASR:
        def transcribe_file(self, audio_path: Path) -> TranscriptResult:
            return TranscriptResult(
                text="متن خام",
                language="fa",
                segments=[{"start": 0.0, "end": 1.0, "text": "متن خام"}],
            )

    class FakeCleaner:
        def clean(self, text: str, timeout: float) -> str:
            raise RuntimeError("ollama down")

    monkeypatch.setattr(
        "persian_live_transcriber.server._convert_audio_to_wav",
        lambda input_path, output_path: _write_silent_wav(output_path),
    )
    monkeypatch.setattr("persian_live_transcriber.server.WhisperASR", FakeASR)
    monkeypatch.setattr("persian_live_transcriber.server.OllamaCleaner", lambda: FakeCleaner())

    client = TestClient(app)
    response = client.post("/api/transcribe-file?cleanup=true", content=b"audio")

    assert response.status_code == 200
    segment = response.json()["cleanSegments"][0]
    assert segment["text"] == "متن خام"
    assert segment["cleanupFailed"] is True


def test_transcript_segments_fallback_uses_full_text_and_duration() -> None:
    result = TranscriptResult(text="یک متن کامل", language=None, segments=[])

    segments = _transcript_segments(result, duration=3.25)

    assert segments == [
        {
            "index": 0,
            "start": 0.0,
            "end": 3.25,
            "startLabel": "00:00:00,000",
            "endLabel": "00:00:03,250",
            "text": "یک متن کامل",
            "language": "fa",
        }
    ]


def test_transcript_segments_accepts_lightning_whisper_list_shape() -> None:
    result = TranscriptResult(text="سلام", language="fa", segments=[[25, 150, "سلام"]])

    segments = _transcript_segments(result, duration=2.0)

    assert segments[0]["start"] == 0.25
    assert segments[0]["end"] == 1.5
    assert segments[0]["text"] == "سلام"


def test_transcript_segments_drops_common_silence_hallucination() -> None:
    result = TranscriptResult(text="موسیقی در اینجا", language="fa", segments=[])

    assert _transcript_segments(result, duration=0.25) == []


def test_asr_wrapper_passes_persian_language(tmp_path: Path) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.language = None

        def transcribe(self, audio_path: str, language: str | None = None) -> dict[str, object]:
            self.language = language
            return {"text": "سلام", "language": language, "segments": []}

    fake = FakeModel()
    asr = WhisperASR()
    asr._model = fake
    result = asr.transcribe_file(tmp_path / "sample.wav")

    assert result.text == "سلام"
    assert fake.language == "fa"


def test_cleanup_task_sends_pending_and_cleaned_text() -> None:
    class FakeWebSocket:
        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        async def send_json(self, message: dict[str, object]) -> None:
            self.messages.append(message)

    class FakeCleaner:
        def clean(self, text: str, timeout: float) -> str:
            return f"{text}!"

    async def run() -> FakeWebSocket:
        websocket = FakeWebSocket()
        payload = {
            "index": 1,
            "start": 0.0,
            "end": 1.0,
            "startLabel": "00:00:00,000",
            "endLabel": "00:00:01,000",
            "text": "متن خام",
            "language": "fa",
        }
        await _cleanup_and_send(websocket, FakeCleaner(), payload, "متن خام", asyncio.Semaphore(1))
        return websocket

    websocket = asyncio.run(run())

    assert websocket.messages[0]["event"] == "cleaned"
    assert websocket.messages[0]["cleanupPending"] is True
    assert websocket.messages[0]["text"] == "متن خام"
    assert websocket.messages[1]["event"] == "cleaned"
    assert websocket.messages[1]["text"] == "متن خام!"
