#!/usr/bin/env bash
# Gen AI Framework - run API server or Streamlit UI
# Run from project root: ./run.sh [ui]
#   ./run.sh      -> API server (uvicorn)
#   ./run.sh ui   -> Streamlit UI

cd "$(dirname "$0")"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-true}"

# Streamlit UI
if [ "$1" = "ui" ] || [ "$1" = "--ui" ]; then
    if command -v uv &> /dev/null; then
        echo "Starting Streamlit UI (uv)..."
        exec uv run streamlit run ui/app.py --server.address "$HOST" --server.port "${STREAMLIT_PORT:-8501}"
    fi
    if [ -d ".venv" ]; then
        . .venv/bin/activate
    fi
    echo "Starting Streamlit UI..."
    exec streamlit run ui/app.py --server.address "$HOST" --server.port "${STREAMLIT_PORT:-8501}"
fi

# API server
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
