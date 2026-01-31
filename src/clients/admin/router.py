"""Admin client API routes."""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.framework.api.deps import get_llm, get_rag
from src.framework.config import get_settings_dep, FrameworkSettings
from src.framework.llm.base import LLMClient
from src.framework.observability.eval import EvalHarness, EvalDatasetItem
from src.framework.rag.base import RAGClient

router = APIRouter(prefix="/admin", tags=["admin"])


class ComponentStatus(BaseModel):
    name: str
    ok: bool
    message: Optional[str] = None


@router.get("/health/components")
def health_components(
    settings: FrameworkSettings = Depends(get_settings_dep),
    llm: LLMClient = Depends(get_llm),
    rag: RAGClient = Depends(get_rag),
):
    """Check status of framework components (LLM, RAG, config)."""
    components = []
    # Config
    components.append(ComponentStatus(name="config", ok=True, message="loaded"))
    # LLM
    try:
        llm.invoke("Say OK in one word.")
        components.append(ComponentStatus(name="llm", ok=True, message="openai ok"))
    except Exception as e:
        components.append(ComponentStatus(name="llm", ok=False, message=str(e)))
    # RAG (Chroma)
    try:
        rag.retrieve("test", top_k=1)
        components.append(ComponentStatus(name="rag", ok=True, message="chroma ok"))
    except Exception as e:
        components.append(ComponentStatus(name="rag", ok=False, message=str(e)))
    return {"components": components}


@router.get("/config")
def get_config(
    settings: FrameworkSettings = Depends(get_settings_dep),
):
    """Return non-sensitive config (e.g. model names, paths)."""
    return {
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "embeddings_provider": settings.EMBEDDINGS_PROVIDER,
        "sentence_transformer_model": settings.SENTENCE_TRANSFORMER_MODEL,
        "chroma_persist_dir": settings.CHROMA_PERSIST_DIR,
        "upload_dir": settings.UPLOAD_DIR,
        "debug": settings.DEBUG,
        "enable_llm_tracing": getattr(settings, "ENABLE_LLM_TRACING", False),
        "chunking_strategy": getattr(settings, "CHUNKING_STRATEGY", "recursive_character"),
        "rag_hybrid_search": getattr(settings, "RAG_HYBRID_SEARCH", False),
        "rag_rerank_top_n": getattr(settings, "RAG_RERANK_TOP_N", 0),
    }


@router.post("/rag/clear")
def rag_clear(
    rag: RAGClient = Depends(get_rag),
):
    """Clear RAG vector store (if supported)."""
    try:
        rag.clear()
        return {"ok": True, "message": "RAG store cleared"}
    except NotImplementedError:
        return {"ok": False, "message": "Clear not supported"}


class EvalRunRequest(BaseModel):
    """Request body for eval run: either dataset_path or inline items."""

    dataset_path: Optional[str] = Field(None, description="Path to JSON/JSONL eval dataset")
    items: Optional[list[dict]] = Field(None, description="Inline list of {question, expected_answer?, expected_keywords?}")


@router.post("/eval/run")
def eval_run(
    body: EvalRunRequest,
    llm: LLMClient = Depends(get_llm),
    rag: RAGClient = Depends(get_rag),
):
    """Run eval harness on a dataset (file path or inline items). Uses RAG + LLM; returns exact_match_rate, keyword_match_rate, latency."""
    if body.dataset_path:
        items = EvalHarness.load_dataset(body.dataset_path)
    elif body.items:
        items = [
            EvalDatasetItem(
                question=row.get("question", ""),
                expected_answer=row.get("expected_answer"),
                expected_keywords=row.get("expected_keywords"),
                metadata=row.get("metadata", {}),
            )
            for row in body.items
        ]
    else:
        return {"error": "Provide dataset_path or items"}
    if not items:
        return {"total": 0, "exact_match_rate": 0.0, "keyword_match_rate": 0.0, "latency_seconds": 0.0, "per_item": []}
    harness = EvalHarness(llm=llm, rag=rag, top_k=4)
    result = harness.run(items)
    return {
        "total": result.total,
        "exact_match": result.exact_match,
        "keyword_match": result.keyword_match,
        "exact_match_rate": result.exact_match_rate,
        "keyword_match_rate": result.keyword_match_rate,
        "latency_seconds": round(result.latency_seconds, 3),
        "per_item": result.per_item,
    }
