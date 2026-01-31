"""Sentence-transformers embeddings provider (local)."""

from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings

from .base import EmbeddingsProvider


class SentenceTransformerEmbeddingsProvider(EmbeddingsProvider):
    """Local embeddings via sentence-transformers (HuggingFace)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)
