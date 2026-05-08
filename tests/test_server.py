import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from persian_live_transcriber.asr import WhisperASR
from persian_live_transcriber.server import _cleanup_and_send, app


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
