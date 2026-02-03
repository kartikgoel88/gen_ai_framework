# Architecture Decision: Adapter vs Agent Responsibilities

## Context

We're using LangChain's `create_agent` as our primary orchestrator. This affects how we separate responsibilities.

## LangChain's Architecture

When using `create_agent`:

```
┌─────────────────┐
│  ChatModel      │  Returns: AIMessage (with optional tool_calls)
│  (Adapter)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  create_agent   │  Expects: AIMessage with tool_calls
│  (LangChain)    │  Does: Validates, executes tools, handles errors
└─────────────────┘
```

## The Key Insight

**LangChain's agent expects `tool_calls` in the AIMessage format.**

If the LLM natively supports tool calling (OpenAI function calling, Anthropic tool use):
- ✅ Adapter uses native support
- ✅ Returns AIMessage with tool_calls
- ✅ LangChain agent handles validation/execution

If the LLM doesn't natively support tool calling:
- ⚠️ Adapter must parse text response to extract tool calls
- ⚠️ Convert to LangChain's tool_calls format
- ✅ LangChain agent handles validation/execution

## What's Acceptable in the Adapter?

### ✅ Format Conversion (OK)
- Converting native tool calls → LangChain format
- Parsing JSON tool calls from text → LangChain format
- This is just data transformation, not interpretation

### ❌ Agent Logic (NOT OK)
- Validating tool calls against registry
- Executing tools
- Error recovery strategies
- Tool selection logic

## Current Implementation

### Adapter (`LangChainLLMAdapter`)
- ✅ Serializes messages
- ✅ Calls LLM (with native tool calling if available)
- ✅ Converts responses to LangChain format
- ✅ Minimal parsing for text-based tool calls (format conversion only)

### LangChain Agent (`create_agent`)
- ✅ Validates tool calls
- ✅ Executes tools
- ✅ Handles errors
- ✅ Manages conversation flow

### Our Agent (`LangChainReActAgent`)
- ✅ Orchestrates the flow
- ✅ Manages tool registry
- ✅ Handles prompt building
- ✅ Event emission
- ✅ Uses LangChain's agent for execution

## Alternative: Custom Agent

If we wanted full control, we could:

1. **Thin Adapter**: Only LLM communication, returns raw text
2. **Custom Agent**: Parses, validates, executes tools ourselves

But since we're using LangChain's `create_agent`, we follow its expectations.

## Recommendation

**Keep the current approach** with these clarifications:

1. **Adapter**: Format conversion is OK (text → tool_calls format)
2. **Adapter**: Validation/execution is NOT OK (agent's job)
3. **Agent**: Use LangChain's agent for tool execution
4. **Agent**: Add our own orchestration layer on top

The `ToolInterpreter` class is useful for:
- Custom agent implementations (if we build our own)
- Testing and debugging
- Cases where we bypass LangChain's agent

But for the main flow using `create_agent`, the adapter's minimal parsing is acceptable.
