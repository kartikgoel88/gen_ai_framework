"""RAG backend dependencies for FastAPI.

This module provides dependency injection functions for RAG clients,
supporting multiple vector stores (Chroma, Pinecone, Weaviate, Qdrant, pgvector).
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from functools import lru_cache

from ..rag.base import RAGClient
from ..rag.chroma_rag import ChromaRAG
from ..rag.reranker import CrossEncoderReranker
from ..embeddings.base import EmbeddingsProvider
from ..embeddings.openai_provider import OpenAIEmbeddingsProvider
from ..embeddings.sentence_transformer_provider import SentenceTransformerEmbeddingsProvider
from ..config import get_settings_dep, FrameworkSettings


@lru_cache
def _get_embeddings_provider(
    provider: str,
    openai_model: str,
    openai_api_key: str | None,
    st_model: str,
) -> EmbeddingsProvider:
    """Create embeddings provider (duplicated here to avoid circular import)."""
    if provider == "sentence_transformers":
        return SentenceTransformerEmbeddingsProvider(model_name=st_model)
    return OpenAIEmbeddingsProvider(model=openai_model, openai_api_key=openai_api_key)


def _get_reranker_if_enabled(rerank_top_n: int) -> CrossEncoderReranker | None:
    """Return a CrossEncoderReranker if rerank_top_n > 0, else None.
    
    Args:
        rerank_top_n: Number of candidates to rerank (0 = disabled)
        
    Returns:
        CrossEncoderReranker instance or None
    """
    if rerank_top_n <= 0:
        return None
    try:
        return CrossEncoderReranker()
    except Exception:
        return None


@lru_cache
def _get_rag_client(
    persist_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    chunking_strategy: str,
    use_hybrid: bool,
    rerank_top_n: int,
    provider: str,
    openai_model: str,
    openai_api_key: str | None,
    st_model: str,
) -> RAGClient:
    """Create ChromaRAG client with caching.
    
    Args:
        persist_dir: ChromaDB persistence directory
        chunk_size: Text chunk size
        chunk_overlap: Chunk overlap size
        chunking_strategy: Chunking strategy name
        use_hybrid: Enable hybrid search (vector + BM25)
        rerank_top_n: Reranking factor (0 = disabled)
        provider: Embeddings provider name
        openai_model: OpenAI model name
        openai_api_key: OpenAI API key
        st_model: SentenceTransformer model name
        
    Returns:
        ChromaRAG instance
    """
    emb = _get_embeddings_provider(provider, openai_model, openai_api_key, st_model)
    reranker = _get_reranker_if_enabled(rerank_top_n) if rerank_top_n > 0 else None
    return ChromaRAG(
        persist_directory=persist_dir,
        embeddings=emb,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunking_strategy=chunking_strategy,
        use_hybrid=use_hybrid,
        reranker=reranker,
        rerank_top_n=rerank_top_n,
    )


def _create_pinecone_rag(settings: FrameworkSettings) -> RAGClient:
    """Create PineconeRAG instance."""
    from ..rag.pinecone_rag import PineconeRAG
    emb = _get_embeddings_provider(
        settings.EMBEDDINGS_PROVIDER,
        settings.EMBEDDING_MODEL,
        settings.OPENAI_API_KEY,
        settings.SENTENCE_TRANSFORMER_MODEL,
    )
    return PineconeRAG(
        embeddings=emb,
        index_name=settings.PINECONE_INDEX_NAME,
        api_key=settings.PINECONE_API_KEY,
        environment=getattr(settings, "PINECONE_ENV", None),
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )


def _create_weaviate_rag(settings: FrameworkSettings) -> RAGClient:
    """Create WeaviateRAG instance."""
    from ..rag.weaviate_rag import WeaviateRAG
    emb = _get_embeddings_provider(
        settings.EMBEDDINGS_PROVIDER,
        settings.EMBEDDING_MODEL,
        settings.OPENAI_API_KEY,
        settings.SENTENCE_TRANSFORMER_MODEL,
    )
    return WeaviateRAG(
        embeddings=emb,
        url=settings.WEAVIATE_URL,
        index_name=getattr(settings, "WEAVIATE_INDEX_NAME", "Chunk"),
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )


def _create_qdrant_rag(settings: FrameworkSettings) -> RAGClient:
    """Create QdrantRAG instance."""
    from ..rag.qdrant_rag import QdrantRAG
    emb = _get_embeddings_provider(
        settings.EMBEDDINGS_PROVIDER,
        settings.EMBEDDING_MODEL,
        settings.OPENAI_API_KEY,
        settings.SENTENCE_TRANSFORMER_MODEL,
    )
    return QdrantRAG(
        embeddings=emb,
        url=settings.QDRANT_URL,
        collection_name=getattr(settings, "QDRANT_COLLECTION", "rag"),
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )


def _create_pgvector_rag(settings: FrameworkSettings) -> RAGClient:
    """Create PgvectorRAG instance."""
    from ..rag.pgvector_rag import PgvectorRAG
    emb = _get_embeddings_provider(
        settings.EMBEDDINGS_PROVIDER,
        settings.EMBEDDING_MODEL,
        settings.OPENAI_API_KEY,
        settings.SENTENCE_TRANSFORMER_MODEL,
    )
    return PgvectorRAG(
        embeddings=emb,
        connection_string=settings.PGVECTOR_CONNECTION_STRING,
        table_name=getattr(settings, "PGVECTOR_TABLE", "rag_embeddings"),
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )


def get_rag(
    settings: Annotated[FrameworkSettings, Depends(get_settings_dep)],
) -> RAGClient:
    """Dependency that returns the configured RAG client.
    
    Supports multiple vector stores:
    - chroma (default): Local ChromaDB
    - pinecone: Pinecone cloud
    - weaviate: Weaviate vector database
    - qdrant: Qdrant vector search
    - pgvector: PostgreSQL with pgvector extension
    
    Selection is based on VECTOR_STORE setting and availability of
    required configuration.
    
    Args:
        settings: Framework settings (injected via FastAPI Depends)
        
    Returns:
        RAGClient instance for the configured vector store
        
    Example:
        ```python
        @app.post("/rag/query")
        def query_rag(question: str, rag: RAGClient = Depends(get_rag)):
            return rag.query(question)
        ```
    """
    store = (getattr(settings, "VECTOR_STORE", None) or "chroma").lower().strip()
    
    # Pinecone
    if store == "pinecone" and getattr(settings, "PINECONE_API_KEY", None) and getattr(settings, "PINECONE_INDEX_NAME", None):
        return _create_pinecone_rag(settings)
    
    # Weaviate
    if store == "weaviate" and getattr(settings, "WEAVIATE_URL", None):
        return _create_weaviate_rag(settings)
    
    # Qdrant
    if store == "qdrant" and getattr(settings, "QDRANT_URL", None):
        return _create_qdrant_rag(settings)
    
    # pgvector
    if store == "pgvector" and getattr(settings, "PGVECTOR_CONNECTION_STRING", None):
        return _create_pgvector_rag(settings)
    
    # Default: Chroma
    return _get_rag_client(
        persist_dir=settings.CHROMA_PERSIST_DIR,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        chunking_strategy=getattr(settings, "CHUNKING_STRATEGY", "recursive_character"),
        use_hybrid=getattr(settings, "RAG_HYBRID_SEARCH", False),
        rerank_top_n=getattr(settings, "RAG_RERANK_TOP_N", 0),
        provider=settings.EMBEDDINGS_PROVIDER,
        openai_model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
        st_model=settings.SENTENCE_TRANSFORMER_MODEL,
    )
