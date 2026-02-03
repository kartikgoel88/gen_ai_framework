# Layer Separation: Adapter vs Agent

## The Problem

Previously, the adapter was doing agent work:
- ❌ Parsing tool calls from text responses
- ❌ Validating tool calls
- ❌ Error recovery for tool calls
- ❌ Tool strategy selection

This violates LangChain's mental model and creates tight coupling.

## LangChain's Mental Model

```
Layer          Responsibility
─────────────────────────────────────────────
ChatModel      "What did the LLM say?"
Agent          "What do we do with that?"
```

### ChatModel (Adapter) Should ONLY:
1. Convert LangChain messages → Provider format
2. Call the LLM (with tools bound if native support exists)
3. Return LangChain messages (with tool_calls if LLM natively returned them)

### Agent Should Handle:
1. Parsing tool calls from text responses (if LLM doesn't natively support)
2. Validating tool calls
3. Executing tools
4. Handling errors
5. Deciding on retry strategies
6. Tool strategy selection

## Refactored Architecture

### `LangChainLLMAdapter` (ChatModel Layer)

**Location**: `src/framework/adapters/langchain_adapter_refactored.py`

**Responsibilities**:
- Serialize messages to provider format
- Call LLM (with native tool calling if available)
- Return raw LLM response

**Does NOT**:
- Parse tool calls from text
- Validate tool calls
- Handle errors
- Select tools

**Example**:
```python
adapter = LangChainLLMAdapter(llm_client)
adapter = adapter.bind_tools(tools)  # Only for native tool calling

# Call LLM
result = adapter._generate(messages)
# Returns: AIMessage with content (and tool_calls if LLM natively returned them)
```

### `ToolInterpreter` (Agent Layer)

**Location**: `src/framework/agents/tool_interpreter.py`

**Responsibilities**:
- Parse tool calls from text responses
- Validate tool calls against registry
- Execute tools
- Handle errors and recovery
- Return ToolMessages

**Example**:
```python
interpreter = ToolInterpreter(tool_registry, parser_type="json")

# Process LLM response
llm_message = AIMessage(content='{"name": "rag_search", "arguments": {"query": "test"}}')
messages = interpreter.process_llm_response(llm_message)
# Returns: [AIMessage with tool_calls, ToolMessage with results]
```

### `LangChainReActAgent` (Agent Layer)

**Responsibilities**:
- Orchestrate LLM calls and tool execution
- Use ToolInterpreter to process responses
- Manage conversation flow
- Handle retries and error recovery

**Flow**:
1. Build messages with PromptBuilder
2. Call adapter (ChatModel) to get LLM response
3. Use ToolInterpreter to process response
4. If tool calls found, execute them and continue
5. Return final response

## Migration Path

### Option 1: Use Refactored Adapter (Recommended)

```python
from src.framework.adapters.langchain_adapter_refactored import LangChainLLMAdapter
from src.framework.agents.tool_interpreter import ToolInterpreter
from src.framework.agents.tool_registry import ToolRegistry

# Thin adapter - just LLM communication
adapter = LangChainLLMAdapter(llm_client)
adapter = adapter.bind_tools(tools)  # Only for native tool calling

# Agent layer - tool interpretation
registry = ToolRegistry()
for tool in tools:
    registry.register(tool)

interpreter = ToolInterpreter(registry)

# Agent orchestrates
llm_response = adapter._generate(messages)
processed = interpreter.process_llm_response(llm_response.generations[0].message)
```

### Option 2: Update Existing Adapter

The existing adapter can be updated to remove agent responsibilities:
- Remove tool call parsing
- Remove validation
- Remove error recovery
- Keep only message serialization and LLM calls

## Benefits

1. **Clear Separation**: Each layer has a single responsibility
2. **Testability**: Can test adapter and agent independently
3. **Flexibility**: Can swap adapters or interpreters without affecting the other
4. **LangChain Compatibility**: Follows LangChain's mental model
5. **Maintainability**: Easier to understand and modify

## When to Mix Responsibilities

Only mix responsibilities if:
- LangChain agents are NOT your primary orchestrator
- You're building a custom agent framework
- You need tight integration for performance reasons

Otherwise, keep them separate!
