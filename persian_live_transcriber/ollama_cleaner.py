from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "gpt-oss:20b"


def build_cleanup_prompt(text: str) -> str:
    return (
        "تو ویراستار معنی‌محور متن پیاده‌سازی‌شده فارسی هستی.\n"
        "اول بررسی کن متن خام از نظر معنایی قابل فهم هست یا نه.\n"
        "اگر متن قابل فهم است، معنی و ترتیب جمله‌ها را حفظ کن و فقط فاصله‌گذاری، نیم‌فاصله‌های واضح، "
        "علائم نگارشی، دستور زبان و خطاهای شنیداری آشکار را اصلاح کن.\n"
        "اگر متن بی‌معنی، گنگ یا از نظر شنیداری خراب است، محافظه‌کارانه حدس بزن منظور گوینده چه بوده "
        "و متن را به جمله‌ای معنی‌دار و روان تبدیل کن.\n"
        "اطلاعات تازه، نام، عدد، تاریخ، تصمیم یا نتیجه‌ای که پشتوانه واضحی در متن خام ندارد اضافه نکن.\n"
        "کلمات و مخفف‌های انگلیسی را با حروف لاتین حفظ کن، مثل API، meeting و deploy، مگر اینکه متن خام "
        "واضحاً معادل فارسی آن‌ها را آورده باشد.\n"
        "خلاصه نکن، توضیح نده، ترجمه نکن، برچسب یا میزان اطمینان ننویس، Markdown اضافه نکن، و فقط متن "
        "پاک‌سازی‌شده را برگردان.\n\n"
        f"متن خام:\n{text.strip()}"
    )


def build_summary_prompt(text: str) -> str:
    return (
        "تو خلاصه‌نویس دقیق یک سشن ترنسکریپت فارسی هستی.\n"
        "فقط بر اساس متن داده‌شده بنویس؛ چیزی اختراع نکن، نتیجه‌گیری اضافه نکن، و معنی را تغییر نده.\n"
        "اگر تصمیم، عدد، نام، تاریخ، یا کار بعدی در متن آمده، همان را دقیق حفظ کن.\n"
        "خروجی فقط فارسی و با Markdown ساده باشد.\n"
        "خلاصه مفصل بساز و این بخش‌ها را داشته باشد:\n"
        "## خلاصه کلی\n"
        "چند پاراگراف فشرده از موضوع و روند صحبت؛ اگر متن کوتاه است، حداقل یک جمله بنویس.\n\n"
        "## نکات اصلی\n"
        "- نکته‌های مهم را به ترتیب اهمیت بنویس؛ اگر نکته‌ای وجود ندارد بنویس: موردی ثبت نشده است.\n\n"
        "## تصمیم‌ها\n"
        "- اگر تصمیمی وجود ندارد بنویس: موردی ثبت نشده است.\n\n"
        "## کارهای بعدی\n"
        "- اگر کار بعدی وجود ندارد بنویس: موردی ثبت نشده است.\n\n"
        f"متن سشن:\n{text.strip()}"
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

    def _generate(self, prompt: str, timeout: float) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
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

        return str(data.get("response") or "").strip()

    def clean(self, text: str, timeout: float = 90.0) -> str:
        raw = text.strip()
        if not raw:
            return ""

        cleaned = self._generate(build_cleanup_prompt(raw), timeout)
        return cleaned or raw

    def summarize(self, text: str, timeout: float = 120.0) -> str:
        raw = text.strip()
        if not raw:
            return ""

        return self._generate(build_summary_prompt(raw), timeout)
