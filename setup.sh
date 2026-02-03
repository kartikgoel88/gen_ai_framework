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

# Create directories used by the app and UI
mkdir -p uploads uploads/ui_uploads uploads/batch_bills uploads/batch_zips uploads/batch_folders
mkdir -p data/chroma_db data/prompts data/feedback data/golden output/batch
echo "Created uploads/ (and subdirs), data/, output/batch/."

# Optional: Tesseract for PDF OCR (scanned/image-only PDFs)
if command -v tesseract &> /dev/null; then
    echo "Tesseract found: $(tesseract --version 2>/dev/null | head -1 || true)"
else
    echo "Optional: Install Tesseract for OCR on scanned PDFs: brew install tesseract (macOS), apt install tesseract-ocr (Linux)."
fi

echo ""
echo "Setup complete."
echo ""
echo "Usage:"
echo "  Run API:    ./run.sh          (or: uv run uvicorn src.main:app --reload)"
echo "  Run UI:     ./run.sh ui       (or: uv run streamlit run ui/app.py)"
echo "  Run CLI:    ./run.sh cli      (or: uv run rag-agent-cli interactive)"
echo "  Run tests:  uv run pytest tests/"
echo ""
echo "CLI Tools:"
echo "  Interactive:  rag-agent-cli interactive  (or: rag-agent-cli i)"
echo "  RAG ops:      rag-agent-cli rag ingest --file doc.txt"
echo "  Agent ops:    rag-agent-cli agent invoke --message 'Hello'"
echo "  List tools:   rag-agent-cli agent tools"
echo ""
echo "  See README.md for full CLI documentation."
echo ""
echo "  API docs:  http://localhost:8000/docs"
