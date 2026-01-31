"""Pinecone RAG backend. Optional: pip install gen-ai-framework[vectorstore-pinecone]."""

from typing import Any, Optional

from ..embeddings.base import EmbeddingsProvider
from .base import RAGClient
from .chunking import get_text_splitter


class PineconeRAG(RAGClient):
    """RAG client using Pinecone vector store."""

    def __init__(
        self,
        embeddings: EmbeddingsProvider,
        index_name: str,
        api_key: str,
        environment: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=api_key)
            self._index = pc.Index(index_name)
        except ImportError as e:
            raise ImportError("Pinecone backend requires: pip install pinecone-client") from e
        self._embeddings = embeddings
        self._index_name = index_name
        self._splitter = get_text_splitter("recursive_character", chunk_size, chunk_overlap)

    def add_documents(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        from uuid import uuid4
        all_vectors = []
        for i, text in enumerate(texts):
            chunks = self._splitter.split_text(text)
            meta = (metadatas or [{}])[i] if (metadatas and i < len(metadatas)) else {}
            for chunk in chunks:
                vec = self._embeddings.embed_query(chunk)
                all_vectors.append({
                    "id": str(uuid4()),
                    "values": vec,
                    "metadata": {**meta, "content": chunk},
                })
        if all_vectors:
            self._index.upsert(vectors=all_vectors)

    def retrieve(self, query: str, top_k: int = 4, **kwargs: Any) -> list[dict[str, Any]]:
        qvec = self._embeddings.embed_query(query)
        result = self._index.query(vector=qvec, top_k=top_k, include_metadata=True, **kwargs)
        out = []
        for match in (result.matches or []):
            meta = (match.metadata or {}).copy()
            content = meta.pop("content", "")
            out.append({"content": content, "metadata": meta})
        return out

    def query(self, question: str, llm_client: Any = None, **kwargs: Any) -> str:
        contexts = self.retrieve(question, **{k: v for k, v in kwargs.items() if k != "llm_client"})
        context_text = "\n\n".join(c["content"] for c in contexts)
        if not llm_client:
            return context_text
        prompt = f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
        return llm_client.invoke(prompt)

    def export_corpus(self, format: str = "jsonl", **kwargs: Any) -> list[dict[str, Any]]:
        """Export all vectors from index (metadata must include 'content'). Uses index stats + fetch if available."""
        try:
            dim = len(self._embeddings.embed_query("x"))
            result = self._index.query(vector=[0.0] * dim, top_k=10000, include_metadata=True)
            out = []
            for match in (result.matches or []):
                meta = (match.metadata or {}).copy()
                content = meta.pop("content", "")
                out.append({"content": content, "metadata": meta})
            return out
        except Exception:
            return []
