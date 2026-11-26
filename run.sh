#!/usr/bin/env bash
# Start the API + the built single-page UI in one process.
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
[[ -x "$PY" ]] || PY="python3"

if [[ ! -f .env ]]; then
  echo "!! .env is missing - copy .env.example and add your API keys." >&2
  exit 1
fi

if [[ ! -f frontend/dist/index.html ]]; then
  echo ".. frontend/dist not found; the setup page will be served at /."
  echo "   Build it with: (cd frontend && npm install && npm run build)"
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
echo ">> http://${HOST}:${PORT}"
exec "$PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT" "$@"
