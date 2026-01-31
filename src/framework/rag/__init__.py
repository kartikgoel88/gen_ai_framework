"""RAG pipeline: vector store, embeddings, retrieval, chunking, reranking."""

from .base import RAGClient
from .chroma_rag import ChromaRAG
from .chunking import get_text_splitter
from .reranker import Reranker, CrossEncoderReranker

__all__ = [
    "RAGClient",
    "ChromaRAG",
    "get_text_splitter",
    "Reranker",
    "CrossEncoderReranker",
]

# Optional backends (import only when used)
def __getattr__(name):
    if name == "PineconeRAG":
        from .pinecone_rag import PineconeRAG
        return PineconeRAG
    if name == "WeaviateRAG":
        from .weaviate_rag import WeaviateRAG
        return WeaviateRAG
    if name == "QdrantRAG":
        from .qdrant_rag import QdrantRAG
        return QdrantRAG
    if name == "PgvectorRAG":
        from .pgvector_rag import PgvectorRAG
        return PgvectorRAG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
