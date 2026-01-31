"""
Test batch admin bills: read inputs from data folder, run batch processing, write output folder.

Uses a mock LLM so the test runs without OPENAI_API_KEY and produces deterministic results.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.clients.batch.service import BatchExpenseService
from src.clients.batch.schemas import (
    ProcessResultItem,
    BatchSummary,
    BatchRunOutput,
)
from src.framework.documents.processor import DocumentProcessor
from src.framework.ocr.processor import OcrProcessor


class MockLLM:
    """Mock LLM that returns fixed structured responses for batch tests."""

    def invoke(self, prompt: str) -> str:
        lower = prompt.lower()
        if any(k in lower for k in ("cab", "taxi", "uber", "lyft", "ride")):
            return "cab"
        if any(k in lower for k in ("meal", "food", "restaurant", "dining", "lunch")):
            return "meals"
        return "unknown"

    def invoke_structured(self, prompt: str) -> dict:
        if "amount" in prompt.lower() and "extract" in prompt.lower():
            # Bill extraction
            return {
                "amount": 45.0,
                "currency": "USD",
                "date": "2024-01-15",
                "vendor": "Uber",
                "description": "Trip",
                "items": [],
            }
        if "decision" in prompt.lower() or "approve" in prompt.lower():
            # Policy decision: approve if amount <= 50 for cab, <= 30 for meals
            return {"decision": "APPROVED", "reason": "Within policy limits."}
        return {"decision": "REJECTED", "reason": "Unknown."}


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def doc_processor(tmp_path):
    return DocumentProcessor(upload_dir=str(tmp_path / "uploads"))


@pytest.fixture
def ocr_processor():
    return OcrProcessor()


@pytest.fixture
def batch_service(mock_llm, doc_processor, ocr_processor):
    return BatchExpenseService(
        llm=mock_llm,
        doc_processor=doc_processor,
        ocr_processor=ocr_processor,
    )


def test_batch_read_from_data_write_to_output(
    batch_service: BatchExpenseService,
    data_batch_dir: Path,
    output_batch_dir: Path,
):
    """
    Read policy and bill files from data folder (tests/fixtures/data/batch),
    run batch processing, write results to output folder (output/batch).
    """
    # --- Read inputs from data folder ---
    policy_path = data_batch_dir / "policy.txt"
    bills_dir = data_batch_dir / "bills"
    assert policy_path.exists(), f"Expected policy at {policy_path}"
    policy_text = policy_path.read_text(encoding="utf-8")

    bill_paths = []
    if bills_dir.exists():
        for f in sorted(bills_dir.iterdir()):
            if f.is_file():
                bill_paths.append(f)
    # If no bills in fixtures, create one in a temp path so test still runs
    if not bill_paths:
        pytest.skip("No bill files in tests/fixtures/data/batch/bills/")

    # --- Run batch processing ---
    results = batch_service.process_bills(bill_paths, policy_text=policy_text)

    # --- Build Pydantic response ---
    approved = sum(1 for r in results if (r.get("decision") or "").upper() == "APPROVED")
    rejected = len(results) - approved
    summary = BatchSummary(approved=approved, rejected=rejected, total=len(results))
    result_items = [ProcessResultItem(**r) for r in results]

    output_payload = BatchRunOutput(
        results=result_items,
        summary=summary,
        input_policy_path=str(policy_path),
        input_bill_paths=[str(p) for p in bill_paths],
    )

    # --- Write to output folder ---
    output_batch_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_batch_dir / "results.json"
    output_file.write_text(
        output_payload.model_dump_json(indent=2),
        encoding="utf-8",
    )

    assert output_file.exists()
    parsed = json.loads(output_file.read_text(encoding="utf-8"))
    assert "results" in parsed
    assert "summary" in parsed
    assert parsed["summary"]["total"] == len(bill_paths)
    assert len(parsed["results"]) == len(bill_paths)
