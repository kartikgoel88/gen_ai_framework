"""
Framework: reusable Gen AI components.

- api: Base API layer (app factory, dependencies, middleware)
- llm: LLM abstraction and providers
- rag: RAG pipeline (vector store, embeddings, retrieval)
- documents: Document processing (extract, parse)
"""

from .config import get_settings, get_settings_dep

__all__ = ["get_settings", "get_settings_dep"]
