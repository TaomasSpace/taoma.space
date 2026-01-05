#!/usr/bin/env bash
set -euo pipefail

# Default backend origin to local FastAPI port
: "${BACKEND_ORIGIN:=http://127.0.0.1:8000}"
export BACKEND_ORIGIN

echo "[render_start] starting FastAPI on 0.0.0.0:8000"
uvicorn main.main:app --host 0.0.0.0 --port 8000 &
UVICORN_PID=$!

echo "[render_start] starting Next.js on 0.0.0.0:${PORT:-3000} (BACKEND_ORIGIN=${BACKEND_ORIGIN})"
cd frontend
npm run start -- -H 0.0.0.0 -p "${PORT:-3000}"

# If Next.js exits, stop uvicorn too
kill "$UVICORN_PID" 2>/dev/null || true
