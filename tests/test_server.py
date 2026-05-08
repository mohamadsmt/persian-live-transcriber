from pathlib import Path

from fastapi.testclient import TestClient

from persian_live_transcriber.asr import WhisperASR
from persian_live_transcriber.server import app


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
