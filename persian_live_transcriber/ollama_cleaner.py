from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "gpt-oss:20b"


def build_cleanup_prompt(text: str) -> str:
    return (
        "تو فقط ویراستار متن پیاده‌سازی‌شده فارسی هستی.\n"
        "وظیفه: فقط فاصله‌گذاری، نیم‌فاصله‌های واضح، علائم نگارشی و خطاهای شنیداری خیلی آشکار "
        "را اصلاح کن.\n"
        "معنی، ترتیب جمله‌ها، عددها، نام‌ها و واژه‌های تخصصی را تغییر نده.\n"
        "خلاصه نکن، توضیح نده، ترجمه نکن، و هیچ متن اضافه‌ای ننویس.\n\n"
        f"متن خام:\n{text.strip()}"
    )


@dataclass(frozen=True)
class OllamaStatus:
    available: bool
    model_available: bool
    error: str | None = None


class OllamaCleaner:
    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def status(self, timeout: float = 2.0) -> OllamaStatus:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            return OllamaStatus(False, False, str(exc))

        names = {item.get("name") for item in data.get("models", [])}
        return OllamaStatus(True, self.model in names)

    def clean(self, text: str, timeout: float = 90.0) -> str:
        raw = text.strip()
        if not raw:
            return ""

        payload = {
            "model": self.model,
            "prompt": build_cleanup_prompt(raw),
            "stream": False,
            "options": {"temperature": 0.0},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))

        cleaned = str(data.get("response") or "").strip()
        return cleaned or raw

