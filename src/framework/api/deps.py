"""FastAPI dependencies for framework components.

This module re-exports all dependency injection functions from focused submodules
for backward compatibility and convenience.

The dependencies are organized into focused modules:
- deps_llm: LLM provider dependencies
- deps_rag: RAG backend dependencies
- deps_embeddings: Embeddings provider dependencies
- deps_documents: Document/OCR processing dependencies
- deps_agents: Agent and chain dependencies
- deps_integrations: External integration dependencies (Confluence, MCP)

All functions are re-exported here to maintain backward compatibility.
"""

# Re-export all dependencies for backward compatibility
from .deps_llm import get_llm
from .deps_rag import get_rag
from .deps_embeddings import get_embeddings
from .deps_documents import (
    get_pdf_ocr_processor,
    get_document_processor,
    get_langchain_loader,
    get_ocr_processor,
    get_docling_processor,
)
from .deps_agents import get_rag_chain, get_agent
from .deps_integrations import get_confluence_client, get_mcp_client

__all__ = [
    # LLM
    "get_llm",
    # RAG
    "get_rag",
    # Embeddings
    "get_embeddings",
    # Documents
    "get_pdf_ocr_processor",
    "get_document_processor",
    "get_langchain_loader",
    "get_ocr_processor",
    "get_docling_processor",
    # Agents & Chains
    "get_rag_chain",
    "get_agent",
    # Integrations
    "get_confluence_client",
    "get_mcp_client",
]
