"""Base API layer: app factory, dependencies, middleware."""

from .app import create_app
from .deps import (
    get_llm,
    get_rag,
    get_rag_chain,
    get_embeddings,
    get_document_processor,
    get_langchain_loader,
    get_ocr_processor,
    get_docling_processor,
    get_mcp_client,
    get_agent,
)

__all__ = [
    "create_app",
    "get_llm",
    "get_rag",
    "get_rag_chain",
    "get_embeddings",
    "get_document_processor",
    "get_langchain_loader",
    "get_ocr_processor",
    "get_docling_processor",
    "get_mcp_client",
    "get_agent",
]
