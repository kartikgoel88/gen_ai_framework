# LangChain Adapter & Agent Improvements

This document describes the comprehensive improvements made to the LangChain adapter and agent implementation.

## Overview

The improvements follow best practices and design patterns to create a more robust, maintainable, and extensible system.

## Key Improvements

### 1. Native Tool Calling Support

**Location**: `src/framework/llm/tool_calling.py`, `src/framework/llm/base.py`

- Added `ToolDefinition`, `ToolCall`, and `ToolCallResponse` data classes
- Extended `LLMClient` interface with `invoke_with_tools()` method
- Providers can now implement native tool calling (e.g., OpenAI function calling, Anthropic tool use)

**Usage**:
```python
from src.framework.llm.tool_calling import ToolDefinition

tool = ToolDefinition(
    name="rag_search",
    description="Search documents",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}}
)

response = llm_client.invoke_with_tools("Search for X", [tool])
if response.has_tool_calls():
    for tool_call in response.tool_calls:
        print(f"Calling {tool_call.name} with {tool_call.arguments}")
```

### 2. Tool Call Parser Strategy Pattern

**Location**: `src/framework/chains/tool_call_parser.py`

- `ToolCallParser` abstract base class
- Multiple implementations:
  - `JSONToolCallParser`: Parses JSON tool calls from text
  - `StructuredOutputParser`: Validates against schema
  - `NativeToolCallParser`: For native SDK formats
- Factory pattern for creating parsers

**Usage**:
```python
from src.framework.chains.tool_call_parser import ToolCallParserFactory

parser = ToolCallParserFactory.create_parser("json")
tool_calls = parser.parse(response_text, available_tools=["rag_search"])
```

### 3. Message Serialization Strategies

**Location**: `src/framework/chains/message_serializer.py`

- `MessageSerializer` abstract base class
- Implementations:
  - `StringSerializer`: Simple string concatenation (default)
  - `JSONSerializer`: JSON format for structured providers
  - `NativeSerializer`: Preserves LangChain message objects
- Factory pattern for creating serializers

**Usage**:
```python
from src.framework.chains.message_serializer import MessageSerializerFactory

serializer = MessageSerializerFactory.create_serializer("json")
prompt = serializer.serialize(messages, tool_definitions)
```

### 4. Error Handling & Recovery

**Location**: `src/framework/chains/errors.py`, `src/framework/chains/error_recovery.py`

- Custom exception hierarchy:
  - `ToolCallError`: Base exception
  - `ToolCallParseError`: Parsing failures
  - `ToolNotFoundError`: Invalid tool names
  - `InvalidToolArgumentsError`: Invalid arguments
- `ErrorRecoveryHandler` interface with implementations:
  - `ToolCallRetryHandler`: Retries with different strategies
  - `ParseErrorHandler`: Fallback handling
  - `RateLimitHandler`: Exponential backoff for rate limits
- `ErrorRecoveryManager`: Coordinates multiple handlers

**Usage**:
```python
from src.framework.chains.error_recovery import ErrorRecoveryManager

recovery_manager = ErrorRecoveryManager()
try:
    result = parse_tool_calls(response)
except ToolCallParseError as e:
    recovered = recovery_manager.handle_error(e, context={"retry_count": 0})
```

### 5. Agent Abstraction Layer

**Location**: `src/framework/agents/executor.py`

- `AgentExecutor` interface for different execution strategies
- `LangChainAgentExecutor`: Implementation using LangChain graphs
- Allows swapping execution backends without changing agent code

### 6. Prompt Builder System

**Location**: `src/framework/agents/prompt_builder.py`

- `PromptBuilder` class for constructing prompts
- Methods: `add_system()`, `add_tools()`, `add_examples()`, `add_chat_history()`
- Builds both LangChain messages and string prompts
- More maintainable than string concatenation

**Usage**:
```python
from src.framework.agents.prompt_builder import PromptBuilder

builder = PromptBuilder()
builder.add_system("You are a helpful assistant")
builder.add_tools(tool_definitions)
builder.add_chat_history(history)
messages = builder.build_messages(user_message)
```

### 7. Tool Registry

**Location**: `src/framework/agents/tool_registry.py`

- `ToolRegistry`: Dynamic tool management
- Register/unregister tools at runtime
- `ToolSelector` strategies:
  - `AllToolsSelector`: All tools
  - `ConditionalToolSelector`: Based on conditions
  - `NamedToolSelector`: Specific tools by name

**Usage**:
```python
from src.framework.agents.tool_registry import ToolRegistry, NamedToolSelector

registry = ToolRegistry()
registry.register(rag_tool)
registry.register(web_search_tool)

selector = NamedToolSelector(["rag_search"])
selected_tools = selector.select(registry)
```

### 8. Provider-Specific Adapters

**Location**: `src/framework/chains/provider_adapters.py`

- Optimized adapters for different providers:
  - `OpenAIAdapter`: JSON serializer, native tool calling
  - `AnthropicAdapter`: JSON serializer, structured parser
  - `GeminiAdapter`: String serializer, JSON parser
  - `GenericAdapter`: Default fallback
- `AdapterFactory`: Creates appropriate adapter based on provider

**Usage**:
```python
from src.framework.chains.provider_adapters import AdapterFactory

adapter = AdapterFactory.create("openai", llm_client, bound_tools=tools)
```

### 9. Event System

**Location**: `src/framework/agents/events.py`

- `AgentEvent` and `AgentEventType` for event types
- `AgentObserver` interface
- Implementations:
  - `DebugObserver`: Logs events for debugging
  - `MetricsObserver`: Collects metrics
- `EventEmitter`: Manages observers and emits events

**Usage**:
```python
from src.framework.agents.events import EventEmitter, DebugObserver, MetricsObserver

emitter = EventEmitter()
emitter.subscribe(DebugObserver())
emitter.subscribe(MetricsObserver())

emitter.emit(AgentEvent(
    event_type=AgentEventType.TOOL_CALL,
    content="Calling rag_search",
    metadata={"tool_name": "rag_search"}
))
```

## Updated Components

### LangChainLLMAdapter

**Improvements**:
- Uses `MessageSerializerFactory` for serialization
- Uses `ToolCallParserFactory` for parsing
- Supports native tool calling via `invoke_with_tools()`
- Error recovery via `ErrorRecoveryManager`
- Configurable serializer/parser types
- Better streaming support with tool call parsing

**New Parameters**:
- `serializer_type`: "string", "json", "native"
- `parser_type`: "json", "structured", "native"
- `use_native_tool_calling`: Enable native tool calling if available

### LangChainReActAgent

**Improvements**:
- Uses `AgentExecutor` abstraction
- Uses `PromptBuilder` for prompt construction
- Uses `ToolRegistry` for tool management
- Uses `EventEmitter` for observability
- Uses `AdapterFactory` for provider-specific adapters

**New Features**:
- Dynamic tool registration
- Event-based observability
- Better prompt management
- Provider-specific optimizations

## Migration Guide

### For Existing Code

The changes are backward compatible. Existing code will continue to work, but you can opt into new features:

```python
# Old way (still works)
agent = create_tool_agent(llm, rag_client=rag)

# New way with provider-specific adapter
from src.framework.chains.provider_adapters import AdapterFactory
adapter = AdapterFactory.create("openai", llm)
agent = LangChainReActAgent(llm=adapter, tools=tools)
```

### Using New Features

```python
# Use tool registry
from src.framework.agents.tool_registry import ToolRegistry
registry = ToolRegistry()
registry.register(tool1)
registry.register(tool2)

# Use prompt builder
from src.framework.agents.prompt_builder import PromptBuilder
builder = PromptBuilder()
builder.add_system("Custom system prompt")
builder.add_tools(tool_definitions)
messages = builder.build_messages("User question")

# Use event system
from src.framework.agents.events import MetricsObserver
observer = MetricsObserver()
agent._event_emitter.subscribe(observer)
# ... use agent ...
metrics = observer.get_metrics()
```

## Benefits

1. **Reliability**: Better error handling and recovery
2. **Maintainability**: Clear separation of concerns
3. **Extensibility**: Easy to add new parsers, serializers, handlers
4. **Performance**: Provider-specific optimizations
5. **Observability**: Event system for debugging and metrics
6. **Flexibility**: Dynamic tool management, configurable strategies

## Future Enhancements

Potential future improvements:
- Plugin system for custom parsers/serializers
- Caching for tool descriptions
- Lazy tool loading
- Message batching support
- More provider-specific optimizations
