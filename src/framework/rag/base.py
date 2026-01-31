"""Abstract RAG client interface."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class RAGClient(ABC):
    """Abstract interface for RAG (retrieval-augmented generation) clients."""

    @abstractmethod
    def add_documents(self, texts: list[str], metadatas: Optional[list[dict]] = None) -> None:
        """Ingest documents into the vector store (chunked internally if needed)."""
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 4, **kwargs: Any) -> list[dict[str, Any]]:
        """Retrieve relevant chunks for a query. Each item has 'content' and optional 'metadata'."""
        ...

    @abstractmethod
    def query(self, question: str, llm_client: Any = None, **kwargs: Any) -> str:
        """Run RAG: retrieve context and (if llm_client provided) generate an answer."""
        ...

    def clear(self) -> None:
        """Optional: clear the store. Override if supported."""
        pass

    def export_corpus(self, format: str = "jsonl", **kwargs: Any) -> list[dict[str, Any]]:
        """Export all chunks/corpus for training or sharing. Returns list of {content, metadata}.
        Override in backends; default returns []."""
        return []
