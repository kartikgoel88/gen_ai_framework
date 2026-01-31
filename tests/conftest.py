"""Pytest fixtures and configuration."""

import pytest
from pathlib import Path

# Project root (parent of tests/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Test data: read inputs from data folder
DATA_BATCH = PROJECT_ROOT / "tests" / "fixtures" / "data" / "batch"
# Output folder: write results here
OUTPUT_DIR = PROJECT_ROOT / "output"
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
    """Path to output/batch. Created if missing."""
    OUTPUT_BATCH.mkdir(parents=True, exist_ok=True)
    return OUTPUT_BATCH
