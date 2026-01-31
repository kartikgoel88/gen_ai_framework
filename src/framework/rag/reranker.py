"""Optional reranker: score (query, doc) and return top_k (e.g. cross-encoder)."""

from abc import ABC, abstractmethod
from typing import Any


class Reranker(ABC):
    """Abstract reranker: score query-document pairs and return top indices."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int,
        content_key: str = "content",
    ) -> list[dict[str, Any]]:
        """Score documents against query and return top_k, each with 'content' and 'metadata'."""
        ...


class CrossEncoderReranker(Reranker):
    """Rerank using a sentence-transformers cross-encoder (query, doc)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name)
        except Exception as e:
            raise ImportError(
                "CrossEncoderReranker requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from e
        self._model_name = model_name

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int,
        content_key: str = "content",
    ) -> list[dict[str, Any]]:
        if not documents:
            return []
        pairs = [(query, d.get(content_key, "") or "") for d in documents]
        scores = self._model.predict(pairs)
        indexed = list(zip(scores, documents))
        indexed.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in indexed[:top_k]]
