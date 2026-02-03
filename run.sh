#!/usr/bin/env bash
# Gen AI Framework - run API server, Streamlit UI, or CLI
# Run from project root: ./run.sh [ui|cli|cli-interactive]
#   ./run.sh              -> API server (uvicorn)
#   ./run.sh ui           -> Streamlit UI
#   ./run.sh cli          -> Show CLI help
#   ./run.sh cli-interactive -> Start interactive CLI mode

cd "$(dirname "$0")"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-true}"

# CLI interactive mode
if [ "$1" = "cli-interactive" ] || [ "$1" = "cli-i" ] || [ "$1" = "interactive" ]; then
    if command -v uv &> /dev/null; then
        echo "Starting interactive CLI (uv)..."
        exec uv run rag-agent-cli interactive "${@:2}"
    fi
    if [ -d ".venv" ]; then
        . .venv/bin/activate
    fi
    echo "Starting interactive CLI..."
    exec rag-agent-cli interactive "${@:2}"
fi

# CLI help
if [ "$1" = "cli" ]; then
    if command -v uv &> /dev/null; then
        echo "CLI Tools (using uv):"
        echo ""
        uv run rag-agent-cli --help
        echo ""
        echo "Quick start:"
        echo "  ./run.sh cli-interactive  - Start interactive mode"
        echo "  rag-agent-cli interactive - Start interactive mode"
        echo "  rag-agent-cli rag ingest --file doc.txt"
        echo "  rag-agent-cli agent invoke --message 'Hello'"
        exit 0
    fi
    if [ -d ".venv" ]; then
        . .venv/bin/activate
    fi
    echo "CLI Tools:"
    echo ""
    rag-agent-cli --help
    echo ""
    echo "Quick start:"
    echo "  ./run.sh cli-interactive  - Start interactive mode"
    echo "  rag-agent-cli interactive - Start interactive mode"
    echo "  rag-agent-cli rag ingest --file doc.txt"
    echo "  rag-agent-cli agent invoke --message 'Hello'"
    exit 0
fi

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
