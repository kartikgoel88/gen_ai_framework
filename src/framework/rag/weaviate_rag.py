"""Weaviate RAG backend. Optional: pip install gen-ai-framework[vectorstore-weaviate]."""

from typing import Any, Optional

from ..embeddings.base import EmbeddingsProvider
from .base import RAGClient
from .chunking import get_text_splitter


class WeaviateRAG(RAGClient):
    """RAG client using Weaviate vector store."""

    def __init__(
        self,
        embeddings: EmbeddingsProvider,
        url: str = "http://localhost:8080",
        index_name: str = "Chunk",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        try:
            import weaviate
            self._client = weaviate.Client(url=url)
        except ImportError as e:
            raise ImportError("Weaviate backend requires: pip install weaviate-client") from e
        self._embeddings = embeddings
        self._class_name = index_name
        self._splitter = get_text_splitter("recursive_character", chunk_size, chunk_overlap)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        if not self._client.schema.contains({"class": self._class_name}):
            self._client.schema.create_class({
                "class": self._class_name,
                "properties": [
                    {"name": "content", "dataType": ["text"]},
                    {"name": "meta", "dataType": ["string"]},
                ],
            })

    def add_documents(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        import json
        for i, text in enumerate(texts):
            chunks = self._splitter.split_text(text)
            meta = (metadatas or [{}])[i] if (metadatas and i < len(metadatas)) else {}
            for chunk in chunks:
                vec = self._embeddings.embed_query(chunk)
                self._client.data_object.create(
                    class_name=self._class_name,
                    data_object={"content": chunk, "meta": json.dumps(meta)},
                    vector=vec,
                )

    def retrieve(self, query: str, top_k: int = 4, **kwargs: Any) -> list[dict[str, Any]]:
        import json
        qvec = self._embeddings.embed_query(query)
        result = self._client.query.get(self._class_name, ["content", "meta"]).with_near_vector({"vector": qvec}).with_limit(top_k).do()
        out = []
        for obj in (result.get("data", {}).get("Get", {}).get(self._class_name) or []):
            content = obj.get("content", "")
            meta = obj.get("meta", "{}")
            try:
                meta = json.loads(meta) if isinstance(meta, str) else meta
            except Exception:
                meta = {}
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
        import json
        try:
            result = self._client.query.get(self._class_name, ["content", "meta"]).with_limit(100_000).do()
            items = result.get("data", {}).get("Get", {}).get(self._class_name) or []
            return [
                {
                    "content": obj.get("content", ""),
                    "metadata": json.loads(obj.get("meta", "{}")) if isinstance(obj.get("meta"), str) else (obj.get("meta") or {}),
                }
                for obj in items
            ]
        except Exception:
            return []
