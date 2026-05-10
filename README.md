# Persian Live Transcriber

A fully local web app for live Persian transcription from a microphone, system audio, or both at the same time.

The app runs a FastAPI server on localhost, serves a browser UI in Persian, transcribes audio with Whisper `large-v3`, and can optionally clean up the raw Persian transcript with a local Ollama model.

## Features

- Live Persian speech-to-text from selected audio input devices.
- Audio source modes for microphone, system audio, or a mixed microphone/system stream.
- Uploaded audio-file transcription through the same Whisper model.
- Whisper multilingual `large-v3` transcription through `lightning-whisper-mlx`.
- Persian language locked to `fa`.
- Optional local text cleanup through Ollama using `gpt-oss:20b`.
- Side-by-side raw and cleaned transcript panes.
- Session output after stopping: unified plain text plus on-demand detailed session summary.
- Copy controls and TXT, JSON, and SRT export.
- Local-only default server bind at `127.0.0.1:8765`.

## Requirements

- macOS with Python `3.11`
- `uv`
- `ffmpeg`
- `lightning-whisper-mlx` compatible hardware/runtime
- Optional: Ollama with the `gpt-oss:20b` model for transcript cleanup
- Optional for macOS system audio capture: `blackhole-2ch`

## Dependency Installation

Install the required system tools with Homebrew:

```bash
brew install uv ffmpeg
```

The Python dependencies are managed by `uv`. Running the app or tests will create/use the local
environment automatically:

```bash
uv run --python 3.11 python -c "import fastapi, sounddevice, lightning_whisper_mlx"
uv run --python 3.11 --extra test pytest
```

Install the optional Ollama cleanup dependency:

```bash
brew install ollama
ollama pull gpt-oss:20b
```

Install BlackHole if you want to capture system audio:

```bash
brew install --cask blackhole-2ch
```

The BlackHole installer is a macOS `.pkg` installer and may ask for your admin password in
Terminal. Reboot after installation; Homebrew's cask usually reports this as required.

After reboot, open Audio MIDI Setup and create a Multi-Output Device:

1. Click `+` and choose `Create Multi-Output Device`.
2. Enable your normal output device, such as `MacBook Pro Speakers`.
3. Enable `BlackHole 2ch`.
4. Set macOS sound output to the new Multi-Output Device.
5. In the app UI, select `BlackHole 2ch` as the system-audio input.

Verify that the app can see BlackHole:

```bash
uv run --python 3.11 python -c "import sounddevice as sd; print(sd.query_devices())"
```

If `BlackHole 2ch` is still missing, reboot once more or restart CoreAudio:

```bash
sudo killall coreaudiod
```

## Run

Start the local server:

```bash
./start-transcriber.sh
```

Open the UI:

```text
http://127.0.0.1:8765
```

Use the live controls to transcribe from microphone/system audio, or choose an audio file in the
file picker and click "ترنسکریپت فایل". File transcription resets the current session output and
fills the same raw, cleaned, unified text, summary, and export panes. Files are converted locally
with `ffmpeg`, so common audio/video containers with audio tracks such as `mp3`, `wav`, `m4a`,
`mp4`, `ogg`, `flac`, and `aac` are supported when your local ffmpeg build can decode them.

The start script sets local cache directories by default:

- `UV_CACHE_DIR=.uv-cache`
- `UV_TOOL_DIR=.uv-tools`
- `HF_HOME=.hf-cache`
- `HF_HUB_DISABLE_XET=1`

## Model Preload

The first transcription can take longer because Whisper may need to download or load the model. To prepare the model before using the UI, run:

```bash
HF_HUB_DISABLE_XET=1 uv run --python 3.11 python scripts/preload_model.py
```

## Ollama Cleanup

Ollama cleanup is optional. The ASR model produces the raw transcript first. When cleanup is enabled and Ollama is available, the app sends each raw Persian segment to `gpt-oss:20b` with instructions to only fix spacing, obvious punctuation, and clear recognition mistakes without changing meaning.

After you stop a session, the UI can also build a unified text output and request a detailed Persian summary of the whole session. The summary is generated only on demand and uses the same local Ollama endpoint.

Expected Ollama endpoint:

```text
http://127.0.0.1:11434
```

Expected model:

```text
gpt-oss:20b
```

Install the optional cleanup model:

```bash
ollama pull gpt-oss:20b
```

If Ollama is unavailable or cleanup times out, the UI keeps the raw transcript available.

## Tests

Run the test suite:

```bash
uv run --python 3.11 --extra test pytest
```

## Project Structure

```text
persian_live_transcriber/
  asr.py              Whisper wrapper and Persian language configuration
  audio.py            Input device discovery, capture, resampling, mixing, WAV writing
  ollama_cleaner.py   Local Ollama cleanup prompt and API client
  server.py           FastAPI routes, WebSocket transcription loop, static UI serving
static/
  index.html          Persian browser UI
  app.js              UI state, WebSocket handling, copy/export actions
  styles.css          Responsive layout and transcript styling
scripts/
  preload_model.py    Whisper model preload helper
tests/
  test_audio.py
  test_ollama_cleaner.py
  test_server.py
```

## API Surface

- `GET /` serves the web UI.
- `GET /api/status` reports runtime dependency, audio, ASR, BlackHole, and Ollama status.
- `GET /api/devices` lists available input devices.
- `POST /api/transcribe-file` transcribes one uploaded audio file from the raw request body.
- `POST /api/summarize` generates an on-demand detailed Persian summary for provided session text.
- `WS /ws/transcribe` streams live transcription events.

File transcription query parameters:

- `cleanup`: `true` or `false`
- `filename`: optional display filename

The request body is the audio file bytes directly, not `multipart/form-data`. The local upload
limit is 500 MB. The server stores the file only in a temporary directory, converts it to 16 kHz
mono WAV with `ffmpeg`, and then transcribes the converted file as one full Whisper job for higher
context quality.

WebSocket query parameters:

- `source`: `mic`, `system`, or `both`
- `mic_device`: optional input device id
- `system_device`: optional input device id
- `cleanup`: `true` or `false`
- `chunk_seconds`: segment duration, from `3` to `20` seconds

## Notes

- The app is designed for local use and binds to localhost by default.
- Installed Ollama models are not used for speech recognition. Ollama is only used for optional Persian text cleanup.
- The transcription language is intentionally fixed to Persian (`fa`).
