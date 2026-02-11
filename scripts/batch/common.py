"""Shared configuration and CLI helpers for batch scripts.

- Common argparse arguments (input, output, model, retries, logging).
- Framework config overrides from CLI/env (no FastAPI).
- Logging setup (level, format, optional file).
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure project root on path when scripts are run as __main__
def _ensure_project_root() -> None:
    if __name__ != "__main__" and "src" not in sys.path:
        return
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_project_root()


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    input_required: bool = False,
    output_required: bool = False,
    include_model: bool = True,
    include_retries: bool = True,
    include_dry_run: bool = True,
) -> None:
    """Add common batch script arguments to an ArgumentParser."""
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=input_required,
        help="Input path (file or directory depending on script).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=output_required,
        help="Output path (file or directory depending on script).",
    )
    if include_model:
        parser.add_argument(
            "--model", "-m",
            type=str,
            default=None,
            help="Override LLM model (e.g. gpt-4o, llama3.2). Uses config default if not set.",
        )
    if include_retries:
        parser.add_argument(
            "--retries",
            type=int,
            default=3,
            metavar="N",
            help="Max retries per item on transient failure (default: 3).",
        )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path. If set, logs are also written to this file.",
    )
    if include_dry_run:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="List what would be processed without running (idempotency/resume info).",
        )


def apply_framework_overrides(
    *,
    model: Optional[str] = None,
    env_overrides: Optional[dict[str, str]] = None,
) -> None:
    """Apply CLI/env overrides to framework config. Call before get_settings().

    Updates os.environ and clears the settings cache so the next get_settings()
    returns values that respect these overrides. Use for --model and any other
    framework keys (e.g. LLM_PROVIDER, CHROMA_PERSIST_DIR).
    """
    from src.framework.config import get_settings

    overrides: dict[str, str] = {}
    if model is not None:
        overrides["LLM_MODEL"] = model
    if env_overrides:
        overrides.update(env_overrides)
    if overrides:
        os.environ.update(overrides)
        get_settings.cache_clear()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Configure root logger for batch scripts. Returns the batch scripts' logger.

    - level: DEBUG | INFO | WARNING | ERROR
    - log_file: if set, add a FileHandler
    - format_string: optional; default includes timestamp, level, name, message
    """
    log = logging.getLogger("scripts.batch")
    log.setLevel(getattr(logging, level.upper(), logging.INFO))

    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%dT%H:%M:%S")

    # Avoid duplicate handlers when script is run multiple times in same process
    if not log.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        log.addHandler(handler)
    if log_file and not any(getattr(h, "baseFilename", "") == str(log_file) for h in log.handlers):
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(formatter)
            log.addHandler(fh)
        except OSError:
            log.warning("Could not open log file %s", log_file)

    return log
