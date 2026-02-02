"""LLM Provider Registry.

This module provides a registry pattern for LLM providers, allowing
dynamic provider selection and easy extension without modifying core code.

The registry pattern eliminates the need for if/else chains in dependency
injection code and makes it easy to add new providers.

Example:
    ```python
    from framework.llm.registry import LLMProviderRegistry
    
    # Register a provider
    @LLMProviderRegistry.register("my_provider")
    def create_my_provider(api_key: str, model: str, temperature: float):
        return MyLLMProvider(api_key=api_key, model=model, temperature=temperature)
    
    # Create a provider instance
    llm = LLMProviderRegistry.create(
        provider="openai",
        api_key="sk-...",
        model="gpt-4",
        temperature=0.7
    )
    ```

Available Providers:
    - openai: OpenAI GPT models (default)
    - grok: xAI Grok models (OpenAI-compatible API)
    - gemini: Google Gemini models
    - huggingface: HuggingFace models
"""

from typing import Callable, Dict, Optional, Any
from .base import LLMClient


class LLMProviderRegistry:
    """Registry for LLM provider factories.
    
    This registry allows providers to self-register, eliminating the need
    for if/else chains in dependency injection code.
    """
    
    _providers: Dict[str, Callable[..., LLMClient]] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """Register a provider factory function.
        
        Args:
            name: Provider name (e.g., "openai", "grok")
            
        Returns:
            Decorator function that registers the factory
            
        Example:
            ```python
            @LLMProviderRegistry.register("openai")
            def create_openai(api_key: str, model: str, temperature: float):
                return OpenAILLMProvider(api_key=api_key, model=model, temperature=temperature)
            ```
        """
        def decorator(factory: Callable[..., LLMClient]) -> Callable[..., LLMClient]:
            cls._providers[name.lower().strip()] = factory
            return factory
        return decorator
    
    @classmethod
    def create(
        cls,
        provider: str,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        **kwargs: Any
    ) -> LLMClient:
        """Create an LLM client instance for the given provider.
        
        Args:
            provider: Provider name (e.g., "openai", "grok")
            api_key: API key for the provider (required for most providers)
            model: Model name/identifier
            temperature: Sampling temperature
            **kwargs: Additional provider-specific arguments
            
        Returns:
            LLMClient instance
            
        Raises:
            ValueError: If provider is not registered or required arguments are missing
            
        Example:
            ```python
            llm = LLMProviderRegistry.create(
                provider="openai",
                api_key="sk-...",
                model="gpt-4",
                temperature=0.7
            )
            ```
        """
        provider = (provider or "openai").lower().strip()
        
        if provider not in cls._providers:
            available = ", ".join(sorted(cls._providers.keys()))
            raise ValueError(
                f"Unknown LLM provider: {provider}. "
                f"Available providers: {available}"
            )
        
        factory = cls._providers[provider]
        return factory(api_key=api_key, model=model, temperature=temperature, **kwargs)
    
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
    """Register built-in LLM providers."""
    from .openai_provider import OpenAILLMProvider
    from .grok_provider import GrokLLMProvider
    from .gemini_provider import GeminiLLMProvider
    from .huggingface_provider import HuggingFaceLLMProvider
    
    @LLMProviderRegistry.register("openai")
    def create_openai(api_key: Optional[str], model: str, temperature: float, **kwargs):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI. Set OPENAI_API_KEY.")
        return OpenAILLMProvider(api_key=api_key, model=model, temperature=temperature)
    
    @LLMProviderRegistry.register("grok")
    def create_grok(api_key: Optional[str], model: str, temperature: float, **kwargs):
        if not api_key:
            raise ValueError("XAI_API_KEY is required for Grok. Set LLM_PROVIDER=grok and XAI_API_KEY.")
        return GrokLLMProvider(api_key=api_key, model=model, temperature=temperature)
    
    @LLMProviderRegistry.register("gemini")
    def create_gemini(api_key: Optional[str], model: str, temperature: float, **kwargs):
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for Gemini. Set LLM_PROVIDER=gemini and GOOGLE_API_KEY.")
        return GeminiLLMProvider(api_key=api_key, model=model, temperature=temperature)
    
    @LLMProviderRegistry.register("huggingface")
    def create_huggingface(api_key: Optional[str], model: str, temperature: float, **kwargs):
        if not api_key:
            raise ValueError(
                "HUGGINGFACE_API_KEY is required for Hugging Face. "
                "Set LLM_PROVIDER=huggingface and HUGGINGFACE_API_KEY."
            )
        return HuggingFaceLLMProvider(api_key=api_key, model=model, temperature=temperature)


# Register providers on module import
_register_builtin_providers()
