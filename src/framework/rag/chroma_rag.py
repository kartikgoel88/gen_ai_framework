"""ChromaDB-based RAG implementation with optional hybrid search and reranking."""

from pathlib import Path
from typing import Any, Optional

from langchain_chroma import Chroma

from ..embeddings.base import EmbeddingsProvider
from .base import RAGClient
from .chunking import get_text_splitter
from .reranker import Reranker


class ChromaRAG(RAGClient):
    """RAG client using ChromaDB and configurable embeddings. Supports chunking strategy, hybrid (vector + BM25) search, and optional reranking."""

    def __init__(
        self,
        persist_directory: str,
        embeddings: EmbeddingsProvider,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunking_strategy: str = "recursive_character",
        use_hybrid: bool = False,
        reranker: Optional[Reranker] = None,
        rerank_top_n: int = 0,
    ):
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self._embeddings = embeddings
        self._persist_dir = persist_directory
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._chunking_strategy = chunking_strategy
        self._use_hybrid = use_hybrid
        self._reranker = reranker
        self._rerank_top_n = max(0, rerank_top_n)
        self._splitter = get_text_splitter(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
        # For hybrid: in-memory list of (text, metadata) in same order as Chroma docs (by _idx)
        self._bm25_texts: list[str] = []
        self._bm25_metadatas: list[dict] = []
        self._bm25_index: Any = None  # BM25Okapi, built lazily

    def _build_bm25(self) -> None:
        if self._bm25_index is not None:
            return
        if not self._bm25_texts and self._use_hybrid:
            # Load from Chroma on restart (persisted store)
            try:
                coll = getattr(self._store, "_collection", None)
                if coll is not None:
                    data = coll.get(include=["documents", "metadatas"])
                    if data and data.get("documents"):
                        self._bm25_texts = list(data["documents"])
                        metas = data.get("metadatas")
                        self._bm25_metadatas = list(metas) if metas else [{} for _ in self._bm25_texts]
            except Exception:
                pass
        if not self._bm25_texts:
            return
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [t.split() for t in self._bm25_texts]
            self._bm25_index = BM25Okapi(tokenized)
        except ImportError:
            self._bm25_index = None

    def add_documents(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        chunks = []
        meta_list: list[dict] = []
        start_idx = len(self._bm25_texts)
        for i, text in enumerate(texts):
            text_chunks = self._splitter.split_text(text)
            meta = (metadatas or [{}])[i] if (metadatas and i < len(metadatas)) else {}
            for j, tc in enumerate(text_chunks):
                m = {**meta, "_idx": start_idx + len(chunks)}
                meta_list.append(m)
                chunks.append(tc)
                if self._use_hybrid:
                    self._bm25_texts.append(tc)
                    self._bm25_metadatas.append(m)
            start_idx = len(self._bm25_texts)
        if chunks:
            self._store.add_texts(texts=chunks, metadatas=meta_list)
            # Note: Chroma 0.4.x+ automatically persists, no need to call persist()
            self._bm25_index = None  # invalidate so next retrieve rebuilds

    def retrieve(self, query: str, top_k: int = 4, **kwargs: Any) -> list[dict[str, Any]]:
        use_hybrid = kwargs.pop("use_hybrid", self._use_hybrid)
        rerank_top_n = kwargs.pop("rerank_top_n", self._rerank_top_n)
        fetch_k = top_k
        if self._reranker and rerank_top_n > 0:
            fetch_k = max(fetch_k, top_k * rerank_top_n)
        if use_hybrid and self._bm25_texts:
            return self._retrieve_hybrid(query, top_k, fetch_k, rerank_top_n, **kwargs)
        docs = self._store.similarity_search(query, k=fetch_k, **kwargs)
        result = [
            {"content": d.page_content, "metadata": getattr(d, "metadata", {})}
            for d in docs
        ]
        if self._reranker and rerank_top_n > 0 and len(result) > top_k:
            result = self._reranker.rerank(query, result, top_k=top_k, content_key="content")
        return result[:top_k]

    def _retrieve_hybrid(
        self,
        query: str,
        top_k: int,
        fetch_k: int,
        rerank_top_n: int,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        self._build_bm25()
        vector_docs = self._store.similarity_search(query, k=fetch_k, **kwargs)
        vector_list = [
            {"content": d.page_content, "metadata": getattr(d, "metadata", {})}
            for d in vector_docs
        ]
        if not self._bm25_index:
            if self._reranker and rerank_top_n > 0 and len(vector_list) > top_k:
                return self._reranker.rerank(query, vector_list, top_k=top_k, content_key="content")
            return vector_list[:top_k]
        tokenized_query = query.split()
        bm25_scores = self._bm25_index.get_scores(tokenized_query)
        top_bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[: fetch_k * 2]
        # RRF merge
        vector_rank_by_idx = {
            d["metadata"].get("_idx", i): (i, d) for i, d in enumerate(vector_list)
        }
        bm25_rank_by_idx = {idx: rank for rank, idx in enumerate(top_bm25_indices)}
        all_indices = set(vector_rank_by_idx) | set(bm25_rank_by_idx)
        k_rrf = 60
        scores: dict[int, float] = {}
        doc_by_idx: dict[int, dict[str, Any]] = {}
        for idx in all_indices:
            rrf = 0.0
            if idx in vector_rank_by_idx:
                rank, doc = vector_rank_by_idx[idx]
                rrf += 1.0 / (k_rrf + rank + 1)
                doc_by_idx[idx] = doc
            if idx in bm25_rank_by_idx:
                rank = bm25_rank_by_idx[idx]
                rrf += 1.0 / (k_rrf + rank + 1)
                if idx not in doc_by_idx and 0 <= idx < len(self._bm25_texts):
                    doc_by_idx[idx] = {
                        "content": self._bm25_texts[idx],
                        "metadata": self._bm25_metadatas[idx],
                    }
            scores[idx] = rrf
        sorted_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        result = [doc_by_idx[idx] for idx in sorted_indices if idx in doc_by_idx]
        if self._reranker and rerank_top_n > 0 and len(result) > top_k:
            result = self._reranker.rerank(query, result, top_k=top_k, content_key="content")
        return result[:top_k]

    def query(self, question: str, llm_client: Any = None, **kwargs: Any) -> str:
        retrieve_kw = {k: v for k, v in kwargs.items() if k != "llm_client"}
        contexts = self.retrieve(question, **retrieve_kw)
        context_text = "\n\n".join(c["content"] for c in contexts)
        if not llm_client:
            return context_text
        prompt = f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
        return llm_client.invoke(prompt)

    def export_corpus(self, format: str = "jsonl", **kwargs: Any) -> list[dict[str, Any]]:
        """Export all chunks (content + metadata) for training or sharing."""
        try:
            coll = getattr(self._store, "_collection", None)
            if coll is None:
                return []
            data = coll.get(include=["documents", "metadatas"])
            if not data or not data.get("documents"):
                return []
            docs = data["documents"]
            metas = data.get("metadatas") or [{}] * len(docs)
            return [{"content": d, "metadata": m} for d, m in zip(docs, metas)]
        except Exception:
            return []
