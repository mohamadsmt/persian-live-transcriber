# Persian Live Transcriber

یک وب‌اپ کاملا لوکال برای ترنسکریپت زنده فارسی از میکروفون، صدای سیستم، یا ترکیب هر دو.

## اجرا

```bash
./start-transcriber.sh
```

بعد از بالا آمدن سرور، UI روی این آدرس در دسترس است:

```text
http://127.0.0.1:8765
```

## وابستگی‌های سیستم

- `python3.11`
- `uv`
- `ffmpeg`
- برای صدای سیستم روی macOS: `blackhole-2ch`

برای نصب BlackHole:

```bash
brew install --cask blackhole-2ch
```

ممکن است بعد از نصب BlackHole نیاز به reboot یا restart صوتی داشته باشی. سپس در Audio MIDI Setup یک Multi-Output Device بساز که هم خروجی اصلی سیستم و هم `BlackHole 2ch` را شامل شود، و در UI همین ابزار `BlackHole 2ch` را به عنوان ورودی صدای سیستم انتخاب کن.

## مدل‌ها

- ASR اصلی: Whisper multilingual `large-v3` از طریق `lightning-whisper-mlx`.
- زبان ترنسکریپت روی `fa` قفل شده است.
- مدل‌های Ollama نصب‌شده ASR نیستند؛ `gpt-oss:20b` فقط برای پاک‌سازی اختیاری متن فارسی استفاده می‌شود.

اولین اجرای ترنسکریپت ممکن است مدل Whisper را دانلود کند.

برای دانلود/آماده‌سازی مدل قبل از استفاده از UI:

```bash
HF_HUB_DISABLE_XET=1 uv run --python 3.11 python scripts/preload_model.py
```

## تست

```bash
uv run --python 3.11 --extra test pytest
```
