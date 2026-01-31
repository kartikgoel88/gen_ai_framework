#!/usr/bin/env bash
# Gen AI Framework - setup script
# Run from project root: ./setup.sh

set -e
cd "$(dirname "$0")"

echo "Setting up Gen AI Framework..."

# Optional: create virtual environment if not already in one
if [ -z "${VIRTUAL_ENV}" ] && [ ! -d ".venv" ]; then
    echo "Creating .venv..."
    python3 -m venv .venv
    echo "Activate with: source .venv/bin/activate"
fi

# Activate venv if present (so pip install goes into it)
if [ -d ".venv" ]; then
    . .venv/bin/activate
fi

# Install project (editable) and dev deps
echo "Installing dependencies..."
pip install --upgrade pip
pip install -e ".[dev]"

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
echo "  Next: source .venv/bin/activate (if not already)"
echo "  Then: ./run.sh  (or: uvicorn src.main:app --reload)"
echo "  Docs: http://localhost:8000/docs"
