"""Standalone batch scripts: direct framework invocation, no HTTP/FastAPI.

Scripts in this package are independently executable via CLI, accept runtime
parameters (input/output paths, model, etc.), use the framework internally,
and follow shared conventions for logging, retry, failure handling, and idempotency.

Example:
    uv run python -m scripts.batch.batch_expense_bills --policy policy.txt --folders ./bills -o out.json
    uv run python -m scripts.batch.batch_rag_ingest --input ./docs --manifest ingested.jsonl
    uv run python -m scripts.batch.batch_rag_queries --input queries.jsonl --output answers.jsonl
"""

from .base import (
    BaseBatchRunner,
    ExitCode,
    append_jsonl,
    existing_output_ids,
    load_jsonl,
    retry_with_backoff,
)
from .common import add_common_args, apply_framework_overrides, setup_logging

__all__ = [
    "BaseBatchRunner",
    "ExitCode",
    "add_common_args",
    "append_jsonl",
    "apply_framework_overrides",
    "existing_output_ids",
    "load_jsonl",
    "retry_with_backoff",
    "setup_logging",
]
