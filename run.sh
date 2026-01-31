#!/usr/bin/env bash
# Gen AI Framework - run the API server
# Run from project root: ./run.sh

cd "$(dirname "$0")"

# Use venv if present
if [ -d ".venv" ]; then
    . .venv/bin/activate
fi

# Default: reload for development
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-true}"

if [ "$RELOAD" = "true" ]; then
    echo "Starting API with reload on http://${HOST}:${PORT}"
    exec uvicorn src.main:app --host "$HOST" --port "$PORT" --reload
else
    echo "Starting API on http://${HOST}:${PORT}"
    exec uvicorn src.main:app --host "$HOST" --port "$PORT"
fi
