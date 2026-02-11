"""Pytest fixtures and configuration."""

import pytest
from pathlib import Path

from src.framework.config import get_settings

# Project root (parent of tests/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Test data: read inputs from data folder
DATA_BATCH = PROJECT_ROOT / "tests" / "fixtures" / "data" / "batch"
# Output folder: from config (relative paths resolved against project root)
_out = get_settings().OUTPUT_DIR
OUTPUT_DIR = (PROJECT_ROOT / _out.lstrip("./")).resolve() if not Path(_out).is_absolute() else Path(_out)
OUTPUT_BATCH = OUTPUT_DIR / "batch"


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def data_batch_dir() -> Path:
    """Path to data/batch (policy + bills)."""
    return DATA_BATCH


@pytest.fixture
def output_batch_dir() -> Path:
    """Path to data/output/batch. Created if missing."""
    OUTPUT_BATCH.mkdir(parents=True, exist_ok=True)
    return OUTPUT_BATCH
