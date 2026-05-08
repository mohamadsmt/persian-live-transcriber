#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
export UV_TOOL_DIR="${UV_TOOL_DIR:-$PWD/.uv-tools}"
export PYTHONUNBUFFERED=1

exec uv run --python 3.11 uvicorn persian_live_transcriber.server:app --host 127.0.0.1 --port 8765

