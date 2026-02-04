"""Provider-specific adapters optimized for different LLM providers."""

from typing import Any, List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from .langchain_adapter import LangChainLLMAdapter
from ..llm.base import LLMClient


class OpenAIAdapter(LangChainLLMAdapter):
    """Adapter optimized for OpenAI providers."""
    
    def __init__(self, llm_client: Any, bound_tools: Optional[List[BaseTool]] = None, **kwargs):
        """
        Initialize OpenAI-optimized adapter.
        
        Uses JSON serializer and parser, enables native tool calling if available.
        """
        super().__init__(
            llm_client=llm_client,
            bound_tools=bound_tools,
            serializer_type="json",
            parser_type="json",
            use_native_tool_calling=True,  # OpenAI supports native function calling
            **kwargs
        )


class AnthropicAdapter(LangChainLLMAdapter):
    """Adapter optimized for Anthropic providers."""
    
    def __init__(self, llm_client: Any, bound_tools: Optional[List[BaseTool]] = None, **kwargs):
        """
        Initialize Anthropic-optimized adapter.
        
        Uses JSON serializer, structured parser for tool use format.
        """
        super().__init__(
            llm_client=llm_client,
            bound_tools=bound_tools,
            serializer_type="json",
            parser_type="structured",
            use_native_tool_calling=True,  # Anthropic supports tool use
            **kwargs
        )


class GeminiAdapter(LangChainLLMAdapter):
    """Adapter optimized for Google Gemini providers."""
    
    def __init__(self, llm_client: Any, bound_tools: Optional[List[BaseTool]] = None, **kwargs):
        """
        Initialize Gemini-optimized adapter.
        
        Uses string serializer, JSON parser.
        """
        super().__init__(
            llm_client=llm_client,
            bound_tools=bound_tools,
            serializer_type="string",
            parser_type="json",
            use_native_tool_calling=False,  # May vary by Gemini version
            **kwargs
        )


class GenericAdapter(LangChainLLMAdapter):
    """Generic adapter for providers without specific optimizations."""
    
    def __init__(self, llm_client: Any, bound_tools: Optional[List[BaseTool]] = None, **kwargs):
        """
        Initialize generic adapter.
        
        Uses default string serializer and JSON parser.
        """
        super().__init__(
            llm_client=llm_client,
            bound_tools=bound_tools,
            serializer_type="string",
            parser_type="json",
            use_native_tool_calling=False,
            **kwargs
        )


class AdapterFactory:
    """Factory for creating provider-specific adapters."""
    
    @staticmethod
    def create(
        provider: str,
        llm_client: LLMClient,
        bound_tools: Optional[List[BaseTool]] = None,
        **kwargs
    ) -> LangChainLLMAdapter:
        """
        Create an adapter optimized for the given provider.
        
        Args:
            provider: Provider name ("openai", "anthropic", "gemini", etc.)
            llm_client: Framework LLMClient instance
            bound_tools: Optional list of tools to bind
            **kwargs: Additional adapter parameters
            
        Returns:
            LangChainLLMAdapter instance
        """
        provider_lower = provider.lower()
        
        adapters = {
            "openai": OpenAIAdapter,
            "anthropic": AnthropicAdapter,
            "claude": AnthropicAdapter,
            "gemini": GeminiAdapter,
            "google": GeminiAdapter,
        }
        
        adapter_class = adapters.get(provider_lower, GenericAdapter)
        return adapter_class(llm_client, bound_tools, **kwargs)
    
    @staticmethod
    def create_from_settings(settings: Any, bound_tools: Optional[List[BaseTool]] = None, **kwargs) -> LangChainLLMAdapter:
        """
        Create adapter from framework settings.
        
        Args:
            settings: FrameworkSettings instance
            bound_tools: Optional list of tools to bind
            **kwargs: Additional adapter parameters
            
        Returns:
            LangChainLLMAdapter instance
        """
        from ..llm.registry import LLMProviderRegistry
        
        provider = getattr(settings, "LLM_PROVIDER", "openai")
        llm_client = LLMProviderRegistry.create(provider, settings)
        
        return AdapterFactory.create(provider, llm_client, bound_tools, **kwargs)
