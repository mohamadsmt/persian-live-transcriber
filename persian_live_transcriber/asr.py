from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "large-v3"
DEFAULT_LANGUAGE = "fa"


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    language: str | None = None
    segments: list[dict[str, Any]] | None = None


def mlx_whisper_available() -> bool:
    return importlib.util.find_spec("lightning_whisper_mlx") is not None


class WhisperASR:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        batch_size: int = 12,
        quant: str | None = None,
    ):
        self.model_name = model_name
        self.language = language
        self.batch_size = batch_size
        self.quant = quant
        self._model = None

    def _load(self):  # noqa: ANN001
        if self._model is None:
            from lightning_whisper_mlx import LightningWhisperMLX  # type: ignore

            self._model = LightningWhisperMLX(
                model=self.model_name,
                batch_size=self.batch_size,
                quant=self.quant,
            )
        return self._model

    def transcribe_file(self, audio_path: Path) -> TranscriptResult:
        model = self._load()
        try:
            result = model.transcribe(audio_path=str(audio_path), language=self.language)
        except TypeError:
            result = model.transcribe(audio_path=str(audio_path))

        return TranscriptResult(
            text=str(result.get("text") or "").strip(),
            language=result.get("language"),
            segments=result.get("segments") or [],
        )
