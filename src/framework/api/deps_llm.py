"""LLM provider dependencies for FastAPI.

This module provides dependency injection functions for LLM clients,
including provider selection, caching, and optional tracing wrapper.
"""

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from ..llm.base import LLMClient
from ..config import get_settings_dep, FrameworkSettings
from ..observability.tracing import TracingLLMClient


def _create_llm_provider(
    provider: str,
    api_key: str | None,
    model: str,
    temperature: float,
    base_url: str | None = None,
) -> LLMClient:
    """Create LLM client for the given provider using the registry pattern.
    
    This function uses the LLMProviderRegistry to create provider instances,
    eliminating if/else chains and making it easy to add new providers.
    
    Args:
        provider: Provider name (openai | grok | gemini | huggingface | local)
        api_key: API key for the provider (optional for local)
        model: Model name/identifier
        temperature: Sampling temperature
        base_url: For provider=local, OpenAI-compatible endpoint URL
        
    Returns:
        LLMClient instance
        
    Raises:
        ValueError: If provider is unknown or API key is missing
    """
    from ..llm.registry import LLMProviderRegistry
    kwargs = {"base_url": base_url} if base_url else {}
    return LLMProviderRegistry.create(
        provider=provider,
        api_key=api_key,
        model=model,
        temperature=temperature,
        **kwargs,
    )


@lru_cache
def _get_llm_cached(
    provider: str,
    api_key: str | None,
    model: str,
    temperature: float,
    base_url: str | None = None,
) -> LLMClient:
    """Cached LLM provider factory.
    
    Uses LRU cache to avoid recreating LLM clients with the same parameters.
    
    Args:
        provider: Provider name
        api_key: API key
        model: Model name
        temperature: Temperature setting
        base_url: For local provider, endpoint URL
        
    Returns:
        Cached LLMClient instance
    """
    return _create_llm_provider(provider, api_key, model, temperature, base_url)


def _get_api_key_for_provider(settings: FrameworkSettings, provider: str) -> str | None:
    """Get the appropriate API key for a given provider.
    
    Args:
        settings: Framework settings
        provider: Provider name (lowercase)
        
    Returns:
        API key or None (local does not require one)
    """
    if provider == "local":
        return None
    if provider == "grok":
        return settings.XAI_API_KEY
    if provider == "gemini":
        return settings.GOOGLE_API_KEY
    if provider == "huggingface":
        return settings.HUGGINGFACE_API_KEY
    return settings.OPENAI_API_KEY


def get_llm(
    settings: Annotated[FrameworkSettings, Depends(get_settings_dep)],
) -> LLMClient:
    """Dependency that returns the configured LLM client.
    
    Supports multiple providers (openai, grok, gemini, huggingface) and
    optionally wraps the client with TracingLLMClient if tracing is enabled.
    
    Args:
        settings: Framework settings (injected via FastAPI Depends)
        
    Returns:
        LLMClient instance, optionally wrapped with TracingLLMClient
        
    Example:
        ```python
        @app.post("/chat")
        def chat(llm: LLMClient = Depends(get_llm)):
            return llm.invoke("Hello!")
        ```
    """
    provider = (settings.LLM_PROVIDER or "openai").lower().strip()
    api_key = _get_api_key_for_provider(settings, provider)
    model = (settings.LLM_LOCAL_MODEL or settings.LLM_MODEL) if provider == "local" else settings.LLM_MODEL
    base_url = getattr(settings, "LLM_LOCAL_BASE_URL", None) if provider == "local" else None

    llm = _get_llm_cached(
        provider=provider,
        api_key=api_key,
        model=model,
        temperature=settings.TEMPERATURE,
        base_url=base_url or None,
    )
    
    # Optionally wrap with tracing
    if getattr(settings, "ENABLE_LLM_TRACING", False):
        level = getattr(logging, (settings.TRACING_LOG_LEVEL or "INFO").upper(), logging.INFO)
        llm = TracingLLMClient(llm, log_level=level)
    
    return llm
