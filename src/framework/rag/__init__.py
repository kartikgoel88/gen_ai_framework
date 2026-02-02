"""RAG pipeline: vector store, embeddings, retrieval, chunking, reranking."""

"""RAG (Retrieval-Augmented Generation) Module.

This module provides RAG functionality with support for multiple vector stores
and retrieval strategies. RAG combines document retrieval with LLM generation
to provide context-aware answers.

Main Components:
    - **RAGClient**: Abstract interface for RAG operations
    - **ChromaRAG**: Default ChromaDB implementation (local, persistent)
    - **PineconeRAG**: Pinecone cloud vector store
    - **WeaviateRAG**: Weaviate vector database
    - **QdrantRAG**: Qdrant vector search engine
    - **PgvectorRAG**: PostgreSQL with pgvector extension
    - **get_text_splitter**: Text chunking utilities
    - **Reranker**: Cross-encoder reranking for improved retrieval

Features:
    - Multiple vector store backends
    - Hybrid search (vector + BM25)
    - Optional reranking for better results
    - Configurable chunking strategies
    - Document metadata support

Example:
    ```python
    from src.framework.rag import ChromaRAG
    from src.framework.embeddings import OpenAIEmbeddingsProvider
    
    # Create embeddings provider
    embeddings = OpenAIEmbeddingsProvider(
        model="text-embedding-3-small",
        openai_api_key="sk-..."
    )
    
    # Create RAG client
    rag = ChromaRAG(
        persist_directory="./data/chroma_db",
        embeddings=embeddings,
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Ingest documents
    rag.add_documents(
        texts=["Document 1 text...", "Document 2 text..."],
        metadatas=[{"source": "doc1.pdf"}, {"source": "doc2.pdf"}]
    )
    
    # Retrieve relevant chunks
    results = rag.retrieve("What is the main topic?", top_k=5)
    
    # Query with LLM (requires LLM client)
    answer = rag.query("What is the main topic?", llm_client=llm)
    ```

Vector Store Selection:
    Set `VECTOR_STORE` environment variable to choose backend:
    - `chroma` (default): Local ChromaDB
    - `pinecone`: Pinecone cloud
    - `weaviate`: Weaviate
    - `qdrant`: Qdrant
    - `pgvector`: PostgreSQL with pgvector
"""

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
    """Lazy import for optional vector store backends.
    
    This allows the module to be imported without requiring all optional
    dependencies (pinecone, weaviate, qdrant, pgvector) to be installed.
    
    Args:
        name: Backend name (PineconeRAG, WeaviateRAG, QdrantRAG, PgvectorRAG)
        
    Returns:
        The requested backend class
        
    Raises:
        AttributeError: If the backend name is not recognized
    """
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
