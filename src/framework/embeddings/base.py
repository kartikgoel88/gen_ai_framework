"""Abstract embeddings interface (LangChain-compatible)."""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingsProvider(ABC):
    """LangChain-style embeddings: embed_documents and embed_query."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents. Returns list of vectors."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query. Returns one vector."""
        ...
