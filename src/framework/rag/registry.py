"""RAG Provider Registry.

This module provides a registry pattern for RAG providers, allowing
dynamic vector store selection and easy extension without modifying core code.

The registry pattern eliminates the need for if/else chains in dependency
injection code and makes it easy to add new vector store backends.

Example:
    ```python
    from framework.rag.registry import RAGProviderRegistry
    
    # Register a provider
    @RAGProviderRegistry.register("my_store")
    def create_my_store(embeddings, **kwargs):
        return MyRAGStore(embeddings=embeddings, **kwargs)
    
    # Create a provider instance
    rag = RAGProviderRegistry.create(
        provider="chroma",
        embeddings=embeddings,
        persist_directory="./data"
    )
    ```

Available Providers:
    - chroma: ChromaDB (default, local)
    - pinecone: Pinecone cloud
    - weaviate: Weaviate vector database
    - qdrant: Qdrant vector search
    - pgvector: PostgreSQL with pgvector extension
"""

from typing import Callable, Dict, Any, Optional
from .base import RAGClient


class RAGProviderRegistry:
    """Registry for RAG provider factories.
    
    This registry allows vector stores to self-register, eliminating the need
    for if/else chains in dependency injection code.
    """
    
    _providers: Dict[str, Callable[..., RAGClient]] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """Register a RAG provider factory function.
        
        Args:
            name: Provider name (e.g., "chroma", "pinecone")
            
        Returns:
            Decorator function that registers the factory
            
        Example:
            ```python
            @RAGProviderRegistry.register("chroma")
            def create_chroma(embeddings, persist_directory, **kwargs):
                return ChromaRAG(embeddings=embeddings, persist_directory=persist_directory, **kwargs)
            ```
        """
        def decorator(factory: Callable[..., RAGClient]) -> Callable[..., RAGClient]:
            cls._providers[name.lower().strip()] = factory
            return factory
        return decorator
    
    @classmethod
    def create(cls, provider: str, **kwargs: Any) -> RAGClient:
        """Create a RAG client instance for the given provider.
        
        Args:
            provider: Provider name (e.g., "chroma", "pinecone")
            **kwargs: Provider-specific arguments (embeddings, persist_directory, etc.)
            
        Returns:
            RAGClient instance
            
        Raises:
            ProviderNotFoundError: If provider is not registered
            
        Example:
            ```python
            rag = RAGProviderRegistry.create(
                provider="chroma",
                embeddings=embeddings,
                persist_directory="./data"
            )
            ```
        """
        provider = (provider or "chroma").lower().strip()
        
        if provider not in cls._providers:
            from ..exceptions import ProviderNotFoundError
            available = sorted(cls._providers.keys())
            raise ProviderNotFoundError(provider, available)
        
        factory = cls._providers[provider]
        return factory(**kwargs)
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names.
        
        Returns:
            List of provider names (sorted)
        """
        return sorted(cls._providers.keys())
    
    @classmethod
    def is_registered(cls, provider: str) -> bool:
        """Check if a provider is registered.
        
        Args:
            provider: Provider name to check
            
        Returns:
            True if provider is registered, False otherwise
        """
        return provider.lower().strip() in cls._providers


# Auto-register built-in providers
def _register_builtin_providers():
    """Register built-in RAG providers."""
    from .chroma_rag import ChromaRAG
    from .pinecone_rag import PineconeRAG
    from .weaviate_rag import WeaviateRAG
    from .qdrant_rag import QdrantRAG
    from .pgvector_rag import PgvectorRAG
    
    @RAGProviderRegistry.register("chroma")
    def create_chroma(embeddings, persist_directory, **kwargs):
        return ChromaRAG(embeddings=embeddings, persist_directory=persist_directory, **kwargs)
    
    @RAGProviderRegistry.register("pinecone")
    def create_pinecone(embeddings, index_name, api_key, **kwargs):
        return PineconeRAG(embeddings=embeddings, index_name=index_name, api_key=api_key, **kwargs)
    
    @RAGProviderRegistry.register("weaviate")
    def create_weaviate(embeddings, url, index_name, **kwargs):
        return WeaviateRAG(embeddings=embeddings, url=url, index_name=index_name, **kwargs)
    
    @RAGProviderRegistry.register("qdrant")
    def create_qdrant(embeddings, url, collection_name, **kwargs):
        return QdrantRAG(embeddings=embeddings, url=url, collection_name=collection_name, **kwargs)
    
    @RAGProviderRegistry.register("pgvector")
    def create_pgvector(embeddings, connection_string, table_name, **kwargs):
        return PgvectorRAG(embeddings=embeddings, connection_string=connection_string, table_name=table_name, **kwargs)


# Register providers on module import
try:
    _register_builtin_providers()
except ImportError:
    # Some providers may not be available if optional dependencies aren't installed
    pass
