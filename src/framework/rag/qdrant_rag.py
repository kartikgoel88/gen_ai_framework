"""Qdrant RAG backend. Optional: pip install gen-ai-framework[vectorstore-qdrant]."""

from typing import Any, Optional

from ..embeddings.base import EmbeddingsProvider
from .base import RAGClient
from .chunking import get_text_splitter


class QdrantRAG(RAGClient):
    """RAG client using Qdrant vector store."""

    def __init__(
        self,
        embeddings: EmbeddingsProvider,
        url: str = "http://localhost:6333",
        collection_name: str = "rag",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            self._client = QdrantClient(url=url)
            self._collection = collection_name
            # Ensure collection exists with correct size (from embeddings)
            dim = len(embeddings.embed_query("test"))
            try:
                self._client.get_collection(collection_name)
            except Exception:
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
        except ImportError as e:
            raise ImportError("Qdrant backend requires: pip install qdrant-client") from e
        self._embeddings = embeddings
        self._splitter = get_text_splitter("recursive_character", chunk_size, chunk_overlap)

    def add_documents(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        from qdrant_client.models import PointStruct
        from uuid import uuid4
        points = []
        for i, text in enumerate(texts):
            chunks = self._splitter.split_text(text)
            meta = (metadatas or [{}])[i] if (metadatas and i < len(metadatas)) else {}
            for chunk in chunks:
                vec = self._embeddings.embed_query(chunk)
                points.append(PointStruct(
                    id=str(uuid4()),
                    vector=vec,
                    payload={"content": chunk, **meta},
                ))
        if points:
            self._client.upsert(collection_name=self._collection, points=points)

    def retrieve(self, query: str, top_k: int = 4, **kwargs: Any) -> list[dict[str, Any]]:
        from qdrant_client.models import Filter
        qvec = self._embeddings.embed_query(query)
        results = self._client.search(
            collection_name=self._collection,
            query_vector=qvec,
            limit=top_k,
            **kwargs,
        )
        out = []
        for hit in results:
            payload = hit.payload or {}
            content = payload.pop("content", "")
            out.append({"content": content, "metadata": payload})
        return out

    def query(self, question: str, llm_client: Any = None, **kwargs: Any) -> str:
        contexts = self.retrieve(question, **{k: v for k, v in kwargs.items() if k != "llm_client"})
        context_text = "\n\n".join(c["content"] for c in contexts)
        if not llm_client:
            return context_text
        prompt = f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
        return llm_client.invoke(prompt)

    def export_corpus(self, format: str = "jsonl", **kwargs: Any) -> list[dict[str, Any]]:
        try:
            records, _ = self._client.scroll(
                collection_name=self._collection,
                limit=100_000,
                with_payload=True,
            )
            return [
                {"content": (r.payload or {}).get("content", ""), "metadata": r.payload or {}}
                for r in records
            ]
        except Exception:
            return []
