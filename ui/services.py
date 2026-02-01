"""UI service layer: build framework and client components without FastAPI Depends."""

from pathlib import Path
from typing import Optional

import streamlit as st

# Framework
from src.framework.config import get_settings
from src.framework.llm.base import LLMClient
from src.framework.documents.processor import DocumentProcessor
from src.framework.documents.pdf_ocr_processor import PdfOcrProcessor
from src.framework.ocr.processor import OcrProcessor
from src.framework.api.deps import (
    get_llm,
    get_document_processor,
    get_ocr_processor,
    get_rag,
    get_agent,
)
from src.clients.batch.service import BatchExpenseService


@st.cache_resource
def _cached_llm():
    """Cached LLM client (uses framework config)."""
    settings = get_settings()
    return get_llm(settings)


@st.cache_resource
def _cached_document_processor():
    """Cached document processor."""
    settings = get_settings()
    pdf_ocr = PdfOcrProcessor(dpi=300, min_text_len=10)
    return get_document_processor(settings, pdf_ocr)


@st.cache_resource
def _cached_ocr_processor():
    """Cached OCR processor."""
    return get_ocr_processor()


@st.cache_resource
def _cached_batch_service():
    """Cached batch expense service (LLM + doc processor + OCR)."""
    llm = _cached_llm()
    doc = _cached_document_processor()
    ocr = _cached_ocr_processor()
    return BatchExpenseService(llm=llm, doc_processor=doc, ocr_processor=ocr)


def get_batch_service() -> BatchExpenseService:
    """Return batch expense service for UI (uses framework + batch client)."""
    return _cached_batch_service()


def get_llm_client() -> LLMClient:
    """Return LLM client for UI (uses framework config)."""
    return _cached_llm()


def get_doc_processor() -> DocumentProcessor:
    """Return document processor for UI."""
    return _cached_document_processor()


def get_rag_client():
    """Return RAG client for UI. May raise if not configured."""
    settings = get_settings()
    return get_rag(settings)


def get_agent_client():
    """Return agent (ReAct + RAG + MCP) for UI. May raise if not configured."""
    settings = get_settings()
    rag = get_rag(settings)
    from src.framework.api.deps import get_mcp_client
    mcp = get_mcp_client(settings)
    return get_agent(settings, rag, mcp)


def save_uploaded_file(uploaded_file, subdir: str = "ui_uploads") -> Optional[Path]:
    """Save a Streamlit UploadedFile to framework upload dir; return path or None."""
    if not uploaded_file:
        return None
    settings = get_settings()
    root = Path(settings.UPLOAD_DIR) / subdir
    root.mkdir(parents=True, exist_ok=True)
    path = root / (uploaded_file.name or "upload")
    path.write_bytes(uploaded_file.getvalue())
    return path
