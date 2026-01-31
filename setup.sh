#!/usr/bin/env bash
# Gen AI Framework - setup script (uses uv)
# Run from project root: ./setup.sh

set -e
cd "$(dirname "$0")"

echo "Setting up Gen AI Framework (uv)..."

# Install/sync dependencies with uv (creates .venv if needed)
if command -v uv &> /dev/null; then
    echo "Syncing dependencies with uv..."
    uv sync
else
    echo "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Copy .env from example if missing
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "  -> Edit .env and set OPENAI_API_KEY (and others) as needed."
else
    echo ".env already exists."
fi

# Create directories used by the app
mkdir -p uploads data/chroma_db data/prompts data/feedback data/golden output/batch
echo "Created uploads/, data/chroma_db/, data/prompts/, data/feedback/, data/golden/, output/batch/."

echo ""
echo "Setup complete."
echo "  Run API: uv run ./run.sh  (or: uv run uvicorn src.main:app --reload)"
echo "  Run tests: uv run pytest tests/"
echo "  Docs: http://localhost:8000/docs"
