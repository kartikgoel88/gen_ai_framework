"""LangChain adapters for integrating framework LLMClient with LangChain.

This module provides adapters that wrap framework LLMClient instances as
LangChain BaseChatModel objects, enabling use in LangChain chains and agents.

Main Components:
    - LangChainLLMAdapter: Main adapter wrapping LLMClient as BaseChatModel
    - MessageSerializer: Serializes LangChain messages to provider format
    - ToolCallParser: Parses tool calls from LLM responses
    - ProviderAdapters: Provider-specific adapter optimizations
    - ErrorRecovery: Error recovery strategies for adapter operations
"""

from .langchain_adapter import LangChainLLMAdapter
from .provider_adapters import (
    AdapterFactory,
    OpenAIAdapter,
    AnthropicAdapter,
    GeminiAdapter,
    GenericAdapter,
)
from .message_serializer import (
    MessageSerializer,
    MessageSerializerFactory,
    StringSerializer,
    JSONSerializer,
    NativeSerializer,
)
from .tool_call_parser import (
    ToolCallParser,
    ToolCallParserFactory,
    JSONToolCallParser,
    StructuredOutputParser,
    NativeToolCallParser,
)
from .tool_utils import (
    convert_tools_to_definitions,
    convert_tool_call_to_langchain_format,
)
from .error_recovery import (
    ErrorRecoveryHandler,
    ErrorRecoveryManager,
    ToolCallRetryHandler,
    ParseErrorHandler,
    RateLimitHandler,
)
from .errors import (
    ToolCallError,
    ToolCallParseError,
    ToolNotFoundError,
    InvalidToolArgumentsError,
)

__all__ = [
    # Main adapter
    "LangChainLLMAdapter",
    # Provider adapters
    "AdapterFactory",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "GenericAdapter",
    # Message serialization
    "MessageSerializer",
    "MessageSerializerFactory",
    "StringSerializer",
    "JSONSerializer",
    "NativeSerializer",
    # Tool call parsing
    "ToolCallParser",
    "ToolCallParserFactory",
    "JSONToolCallParser",
    "StructuredOutputParser",
    "NativeToolCallParser",
    # Utilities
    "convert_tools_to_definitions",
    "convert_tool_call_to_langchain_format",
    # Error recovery
    "ErrorRecoveryHandler",
    "ErrorRecoveryManager",
    "ToolCallRetryHandler",
    "ParseErrorHandler",
    "RateLimitHandler",
    # Errors
    "ToolCallError",
    "ToolCallParseError",
    "ToolNotFoundError",
    "InvalidToolArgumentsError",
]
