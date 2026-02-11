"""Base batch runner: logging, retry, failure handling, exit codes, idempotency.

All batch scripts should use this abstraction for consistent behavior on
large datasets: structured logging, retries with backoff, clear exit codes,
and optional idempotency (resume by skipping already-processed items).
"""

import argparse
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from src.framework.utils.retry_utils import (
    compute_backoff_delay,
    is_rate_limit_error,
)

T = TypeVar("T")

# Exit codes: 0 success, 1 usage/validation, 2 partial failure, 3 total failure
class ExitCode:
    SUCCESS = 0
    USAGE_OR_VALIDATION = 1
    PARTIAL_FAILURE = 2
    TOTAL_FAILURE = 3


__all__ = [
    "ExitCode",
    "BaseBatchRunner",
    "retry_with_backoff",
    "load_jsonl",
    "append_jsonl",
    "existing_output_ids",
]


def retry_with_backoff(
    fn: Callable[[], T],
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    factor: float = 2.0,
    max_delay: float = 60.0,
    log: Optional[logging.Logger] = None,
    is_retryable: Optional[Callable[[Exception], bool]] = None,
) -> T:
    """Run fn(); on failure, retry with exponential backoff. Raise last exception if all retries fail.

    is_retryable(e): if provided, only retry when True (default: retry on rate limit, else any).
    """
    logger = log or logging.getLogger("scripts.batch")
    last_exc: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if is_retryable is not None and not is_retryable(e):
                raise
            if not is_rate_limit_error(e) and is_retryable is None:
                # Default: only retry on rate-limit-like errors
                raise
            if attempt == max_attempts - 1:
                raise
            delay = compute_backoff_delay(
                attempt,
                initial_delay=initial_delay,
                factor=factor,
                max_delay=max_delay,
                rate_limit_min=60.0 if is_rate_limit_error(e) else None,
            )
            logger.warning(
                "Attempt %s/%s failed (%s). Retrying in %.1fs: %s",
                attempt + 1,
                max_attempts,
                type(e).__name__,
                delay,
                str(e)[:200],
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file; return list of dicts. Empty file -> []."""
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object as a single line to path. Creates parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def existing_output_ids(output_path: Path, id_key: str = "id") -> set[str]:
    """For idempotency: return set of ids already present in JSONL output."""
    if not output_path.exists():
        return set()
    ids: set[str] = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if id_key in obj:
                    ids.add(str(obj[id_key]))
            except json.JSONDecodeError:
                continue
    return ids


class BaseBatchRunner(ABC):
    """Base class for batch scripts: setup, run loop, exit codes, optional idempotency.

    Subclasses implement:
      - parse_args() -> namespace
      - validate_args(args) -> None (raise on invalid)
      - run_batch(args) -> dict with keys like: success_count, failure_count, errors, result
    """

    def __init__(self, description: str, logger: Optional[logging.Logger] = None):
        self.description = description
        self.log = logger or logging.getLogger(f"scripts.batch.{self.__class__.__name__}")

    @abstractmethod
    def parse_args(self) -> Any:
        """Build and parse argparse. Return parsed namespace."""
        ...

    @abstractmethod
    def validate_args(self, args: Any) -> None:
        """Validate paths and options. Raise ValueError or FileNotFoundError on invalid."""
        ...

    @abstractmethod
    def run_batch(self, args: Any) -> dict[str, Any]:
        """Execute the batch job. Return dict with success_count, failure_count, errors, result (or similar)."""
        ...

    def run(self) -> int:
        """Main entry: parse args, setup logging, validate, run_batch, then exit with appropriate code."""
        parser = argparse.ArgumentParser(
            description=self.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self.add_arguments(parser)
        args = parser.parse_args()

        from .common import setup_logging

        log_level = getattr(args, "log_level", "INFO")
        log_file = getattr(args, "log_file", None)
        setup_logging(level=log_level, log_file=log_file)
        self.log = logging.getLogger(f"scripts.batch.{self.__class__.__name__}")

        try:
            self.validate_args(args)
        except (ValueError, FileNotFoundError, NotADirectoryError) as e:
            self.log.error("Validation failed: %s", e)
            return ExitCode.USAGE_OR_VALIDATION

        dry_run = getattr(args, "dry_run", False)
        if dry_run:
            self.log.info("Dry run: skipping execution")
            self.dry_run(args)
            return ExitCode.SUCCESS

        try:
            out = self.run_batch(args)
        except Exception as e:
            self.log.exception("Batch run failed: %s", e)
            return ExitCode.TOTAL_FAILURE

        success = out.get("success_count", 0)
        failure = out.get("failure_count", 0)
        errors = out.get("errors", [])

        if failure == 0 and not errors:
            self.log.info("Batch completed successfully. success_count=%s", success)
            return ExitCode.SUCCESS
        if success > 0:
            self.log.warning(
                "Batch completed with partial failure. success=%s failure=%s errors=%s",
                success,
                failure,
                len(errors),
            )
            return ExitCode.PARTIAL_FAILURE
        self.log.error("Batch failed entirely. failure_count=%s", failure)
        return ExitCode.TOTAL_FAILURE

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Override to add script-specific arguments. Default: only --log-level and --log-file."""
        parser.add_argument(
            "--log-level",
            choices=("DEBUG", "INFO", "WARNING", "ERROR"),
            default="INFO",
            help="Logging level.",
        )
        parser.add_argument(
            "--log-file",
            type=Path,
            default=None,
            help="Optional log file path.",
        )

    def dry_run(self, args: Any) -> None:
        """Override to implement dry-run behavior (e.g. list items to process). Default: no-op."""
        self.log.info("Dry run not implemented for this script.")
