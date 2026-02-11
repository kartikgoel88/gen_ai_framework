# Standalone Batch Scripts

Production-grade batch scripts that **directly invoke the framework** (no HTTP, no FastAPI, no REST clients). Each script is independently executable via CLI, supports large datasets, and shares common patterns for logging, retry, failure handling, and idempotency.

**Batch expense bills:** For a documented high-level flow (policy → folders → per-bill extraction, validation, policy evaluation) and how **OCR is used for PDF and image bills**, see [docs/BATCH_EXPENSE_BILLS_FLOW.md](../docs/BATCH_EXPENSE_BILLS_FLOW.md).

## Requirements

- **No HTTP layer** — direct Python calls into `src.framework` and `src.clients.batch`.
- **No FastAPI** — scripts use `get_settings()`, `get_rag()`, `get_llm()`, etc., without a running server.
- **CLI** — argparse-based; all parameters (input, output, model, retries) are runtime arguments.
- **Logging** — configurable level and optional log file.
- **Retry and failure handling** — exponential backoff for rate limits / transient errors; clear exit codes.
- **Idempotent** — safe to re-run; resume where possible (e.g. skip already-processed items).

## Folder Structure

```
scripts/
  batch/
    __init__.py          # Package exports
    base.py              # BaseBatchRunner, retry_with_backoff, exit codes, JSONL helpers
    common.py            # add_common_args, apply_framework_overrides, setup_logging
    batch_expense_bills.py   # Policy + folders/ZIP → expense results JSON
    batch_rag_ingest.py      # Directory of docs → RAG vector store (optional manifest for resume)
    batch_rag_queries.py     # Queries JSONL → RAG answers JSONL (resume by id)
    README.md            # This file
```

## Base Batch Runner

- **`BaseBatchRunner`** (in `base.py`): abstract base with `parse_args()`, `validate_args()`, `run_batch()`, and optional `dry_run()`. Handles logging setup, validation failures, and exit codes.
- **Exit codes**: `0` success, `1` usage/validation error, `2` partial failure, `3` total failure.
- **Retry**: `retry_with_backoff(fn, max_attempts=3, ...)` with exponential backoff; optional `is_retryable` (default: rate-limit-like errors).
- **Idempotency helpers**: `load_jsonl`, `append_jsonl`, `existing_output_ids` for resume-by-id or manifest-based skip.

## Shared Configuration

- **Framework overrides**: `apply_framework_overrides(model=..., env_overrides={...})` updates `os.environ` and clears `get_settings()` cache so the next call uses CLI/env values (e.g. `--model gpt-4o`).
- **Common CLI args**: `add_common_args(parser, ...)` can add `--input`, `--output`, `--model`, `--retries`, `--log-level`, `--log-file`, `--dry-run` where needed.
- Config is read from `.env` and environment; override at runtime via CLI and `apply_framework_overrides()`.

## Example 1: Batch Expense Bills

Process policy + employee folders (or a ZIP of folders) and write a single results JSON.

**Text extraction:** **OCR is used for PDFs and images.** PDFs are processed with `DocumentProcessor(pdf_processor=OcrProcessor)`: image-only or scanned PDFs use OCR (PyMuPDF + Tesseract); native PDF text is used as fallback. Image bill files (e.g. .jpg, .png) use OCR (EasyOCR or Tesseract). See [Batch Expense Bills — High-Level Flow](../docs/BATCH_EXPENSE_BILLS_FLOW.md) for the full flow and OCR details.

```bash
# From project root (script uses CONFIG; see batch_expense_bills.py)
uv run python -m scripts.batch.batch_expense_bills
```

Legacy CLI examples (if the script is run with argparse):

```bash
uv run python -m scripts.batch.batch_expense_bills --policy policy.txt --folders ./bills -o results.json
uv run python -m scripts.batch.batch_expense_bills --policy policy.pdf --zip folders.zip -o results.json --model gpt-4o --retries 5
uv run python -m scripts.batch.batch_expense_bills --policy policy.txt --folders ./bills -o out.json --dry-run
```

- **Idempotent**: Same input → same output; re-run overwrites output file.
- **Retry**: Whole run is retried on rate-limit (429) up to `--retries` with backoff.

## Example 2: Batch RAG Ingest

Ingest all documents from a directory into the RAG vector store. Optional manifest file for resume (skip already-ingested paths).

```bash
uv run python -m scripts.batch.batch_rag_ingest --input ./docs --manifest ingested.jsonl
uv run python -m scripts.batch.batch_rag_ingest --input ./docs --persist-dir ./data/chroma_db --extensions .txt,.md,.pdf --retries 3
uv run python -m scripts.batch.batch_rag_ingest --input ./docs --dry-run
```

- **Idempotent**: Use `--manifest ingested.jsonl`; script appends each ingested path and skips paths already in the manifest on next run.
- **Large datasets**: Processes file-by-file; failures are logged and counted; partial success returns exit code 2.

## Example 3: Batch RAG Queries

Run a list of queries (JSONL) through RAG and append answers to an output JSONL. Skips query IDs already present in the output (resume).

```bash
# queries.jsonl: {"id": "1", "query": "What is X?"}
uv run python -m scripts.batch.batch_rag_queries --input queries.jsonl --output answers.jsonl
uv run python -m scripts.batch.batch_rag_queries --input q.jsonl -o out.jsonl --model gpt-4o --top-k 6 --retries 3
uv run python -m scripts.batch.batch_rag_queries --input q.jsonl -o out.jsonl --dry-run
```

- **Idempotent**: Output is append-only; already-done IDs are skipped on re-run.
- **Input format**: Each line `{"id": "...", "query": "..."}` or `"question"` instead of `"query"`.
- **Output format**: Each line `{"id", "query", "answer"}` or `{"id", "query", "answer", "error"}` on failure.

## Logging

- Set level: `--log-level DEBUG` (or INFO, WARNING, ERROR).
- Optional file: `--log-file batch.log` (logs to stderr and file).
- Format: `timestamp | level | logger_name | message`.

## Error Handling and Exit Codes

| Code | Meaning |
|------|--------|
| 0 | Success |
| 1 | Usage or validation error (e.g. missing file) |
| 2 | Partial failure (some items failed) |
| 3 | Total failure (e.g. exception in run_batch) |

## Avoiding Code Duplication

- **Base class**: All three scripts extend `BaseBatchRunner` and implement `add_arguments`, `parse_args`, `validate_args`, `run_batch`, and optionally `dry_run`.
- **Common helpers**: `add_common_args`, `apply_framework_overrides`, `setup_logging` in `common.py`; `retry_with_backoff`, `load_jsonl`, `append_jsonl`, `existing_output_ids` in `base.py`.
- **Framework access**: Scripts use the same entry points as the rest of the app: `get_settings()`, `get_rag(settings)`, `get_llm(settings)`, `DocumentProcessor()`, and `run()` from `src.clients.batch.runner` (expense). No duplicate service logic.

## Running From Project Root

Always run from the repository root so that `src` and `scripts` are on the path:

```bash
cd /path/to/gen_ai_framework
uv run python -m scripts.batch.batch_expense_bills ...
uv run python -m scripts.batch.batch_rag_ingest ...
uv run python -m scripts.batch.batch_rag_queries ...
```

Or add the project root to `PYTHONPATH` and run the script file directly.
