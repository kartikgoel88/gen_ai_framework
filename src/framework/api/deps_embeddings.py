"""Embeddings provider dependencies for FastAPI.

This module provides dependency injection functions for embeddings providers,
supporting OpenAI and SentenceTransformer backends.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

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
    """Create embeddings provider with caching.
    
    Args:
        provider: Provider name ("openai" or "sentence_transformers")
        openai_model: OpenAI model name
        openai_api_key: OpenAI API key
        st_model: SentenceTransformer model name
        
    Returns:
        EmbeddingsProvider instance
    """
    if provider == "sentence_transformers":
        return SentenceTransformerEmbeddingsProvider(model_name=st_model)
    return OpenAIEmbeddingsProvider(model=openai_model, openai_api_key=openai_api_key)


def get_embeddings(
    settings: Annotated[FrameworkSettings, Depends(get_settings_dep)],
) -> EmbeddingsProvider:
    """Dependency that returns the configured embeddings provider.
    
    Supports OpenAI and SentenceTransformer providers based on
    EMBEDDINGS_PROVIDER setting.
    
    Args:
        settings: Framework settings (injected via FastAPI Depends)
        
    Returns:
        EmbeddingsProvider instance
        
    Example:
        ```python
        @app.post("/embed")
        def embed(text: str, embeddings: EmbeddingsProvider = Depends(get_embeddings)):
            return embeddings.embed(text)
        ```
    """
    return _get_embeddings_provider(
        provider=settings.EMBEDDINGS_PROVIDER,
        openai_model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
        st_model=settings.SENTENCE_TRANSFORMER_MODEL,
    )
