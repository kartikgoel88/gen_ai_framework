"""Evaluation API: golden datasets (regression), feedback store, RAG export."""

from typing import Any, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.framework.api.deps import get_llm, get_rag
from src.framework.config import get_settings_dep, FrameworkSettings
from src.framework.llm.base import LLMClient
from src.framework.rag.base import RAGClient
from src.framework.evaluation.golden import GoldenDatasetRunner, GoldenItem, GoldenRunResult
from src.framework.evaluation.feedback_store import FeedbackStore, FeedbackEntry

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


def _feedback_store(settings: FrameworkSettings = Depends(get_settings_dep)) -> FeedbackStore:
    return FeedbackStore(path=getattr(settings, "FEEDBACK_STORE_PATH", "./data/feedback/feedback.jsonl"))


# --- Golden datasets (regression) ---


class GoldenRunRequest(BaseModel):
    dataset_path: Optional[str] = Field(None, description="Path to JSON/JSONL golden dataset")
    items: Optional[list[dict[str, Any]]] = Field(None, description="Inline items: [{id, inputs, expected_output?, expected_keywords?}]")
    target: str = Field("rag", description="rag | batch | agent (run_fn is inferred from target)")
    compare_mode: str = Field("keyword", description="exact | keyword")


@router.post("/golden/run")
def golden_run(
    body: GoldenRunRequest,
    llm: LLMClient = Depends(get_llm),
    rag: RAGClient = Depends(get_rag),
):
    """Run golden dataset: fixed inputs -> run target (rag/batch/agent) -> compare outputs (regression)."""
    if body.dataset_path:
        items = GoldenDatasetRunner.load_dataset(body.dataset_path)
    elif body.items:
        items = [
            GoldenItem(
                id=row.get("id", str(i)),
                inputs=row.get("inputs", {}),
                expected_output=row.get("expected_output"),
                expected_keywords=row.get("expected_keywords"),
                metadata=row.get("metadata", {}),
            )
            for i, row in enumerate(body.items)
        ]
    else:
        return {"error": "Provide dataset_path or items"}
    if not items:
        return {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0, "per_item": []}

    target = (body.target or "rag").lower()
    if target == "rag":
        def run_fn(inp):
            q = inp.get("query") or inp.get("question") or ""
            return rag.query(q, llm_client=llm)
    elif target == "batch":
        from src.clients.batch.service import BatchExpenseService
        from src.framework.api.deps import get_document_processor, get_pdf_ocr_processor, get_ocr_processor
        _settings = get_settings_dep()
        doc = get_document_processor(_settings, get_pdf_ocr_processor())
        ocr = get_ocr_processor()
        svc = BatchExpenseService(llm=llm, doc_processor=doc, ocr_processor=ocr)
        def run_fn(inp):
            texts = inp.get("bill_texts", [])
            policy = inp.get("policy_text", "")
            names = inp.get("file_names")
            results = svc.process_bills_from_texts(texts, policy, names)
            return results
    else:
        from src.framework.api.deps import get_agent, get_mcp_client
        settings = get_settings_dep()
        _settings = get_settings_dep()
        agent = get_agent(_settings, rag, get_mcp_client(_settings))
        def run_fn(inp):
            msg = inp.get("message") or inp.get("input") or ""
            return agent.invoke(msg)

    runner = GoldenDatasetRunner(run_fn=run_fn, compare_mode=body.compare_mode or "keyword")
    result = runner.run(items)
    return {
        "total": result.total,
        "passed": result.passed,
        "failed": result.failed,
        "pass_rate": result.pass_rate,
        "latency_seconds": round(result.latency_seconds, 3),
        "compare_mode": result.compare_mode,
        "per_item": result.per_item,
    }


# --- Feedback store ---


class FeedbackSubmitRequest(BaseModel):
    prompt: str
    response: str
    feedback: Optional[str] = None
    score: Optional[float] = None
    session_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


@router.post("/feedback")
def feedback_submit(
    body: FeedbackSubmitRequest,
    store: FeedbackStore = Depends(_feedback_store),
):
    """Submit user feedback and model output for fine-tuning or prompt tuning."""
    entry_id = store.add(
        prompt=body.prompt,
        response=body.response,
        feedback=body.feedback,
        score=body.score,
        session_id=body.session_id,
        metadata=body.metadata,
    )
    return {"id": entry_id}


@router.get("/feedback")
def feedback_list(
    limit: int = 100,
    session_id: Optional[str] = None,
    has_feedback: Optional[bool] = None,
    store: FeedbackStore = Depends(_feedback_store),
):
    """List recent feedback entries."""
    entries = store.list_entries(limit=limit, session_id=session_id, has_feedback=has_feedback)
    return {"entries": [e.to_dict() for e in entries]}


@router.post("/feedback/export")
def feedback_export(
    path: Optional[str] = None,
    only_with_feedback: bool = True,
    store: FeedbackStore = Depends(_feedback_store),
):
    """Export feedback entries for fine-tuning (JSONL messages format)."""
    out_path = store.export_for_finetuning(path=path, only_with_feedback=only_with_feedback)
    return {"path": out_path}


# --- RAG export ---


@router.get("/rag/export")
def rag_export(
    rag: RAGClient = Depends(get_rag),
    format: str = "json",
):
    """Export RAG corpus (all chunks) for training or sharing."""
    chunks = rag.export_corpus(format=format)
    return {"count": len(chunks), "chunks": chunks}


@router.get("/rag/export/download")
def rag_export_download(
    rag: RAGClient = Depends(get_rag),
    format: str = "jsonl",
):
    """Download RAG corpus as JSONL file."""
    chunks = rag.export_corpus(format=format)
    import json
    def gen():
        for c in chunks:
            yield json.dumps(c, ensure_ascii=False) + "\n"
    return StreamingResponse(gen(), media_type="application/x-ndjson", headers={"Content-Disposition": "attachment; filename=rag_corpus.jsonl"})
