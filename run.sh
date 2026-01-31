#!/usr/bin/env bash
# Gen AI Framework - run the API server
# Run from project root: ./run.sh

cd "$(dirname "$0")"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-true}"

# Prefer uv if available (matches setup.sh / test_setup.sh)
if command -v uv &> /dev/null; then
    if [ "$RELOAD" = "true" ]; then
        echo "Starting API with reload on http://${HOST}:${PORT} (uv)"
        exec uv run uvicorn src.main:app --host "$HOST" --port "$PORT" --reload
    else
        echo "Starting API on http://${HOST}:${PORT} (uv)"
        exec uv run uvicorn src.main:app --host "$HOST" --port "$PORT"
    fi
fi

# Fallback: use .venv if present
if [ -d ".venv" ]; then
    . .venv/bin/activate
fi

if [ "$RELOAD" = "true" ]; then
    echo "Starting API with reload on http://${HOST}:${PORT}"
    exec uvicorn src.main:app --host "$HOST" --port "$PORT" --reload
else
    echo "Starting API on http://${HOST}:${PORT}"
    exec uvicorn src.main:app --host "$HOST" --port "$PORT"
fi
