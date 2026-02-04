"""Tests for adapters: LangChain adapter, message serialization, tool call parsing."""

from unittest.mock import MagicMock

import pytest

from src.framework.adapters import (
    LangChainLLMAdapter,
    AdapterFactory,
    MessageSerializerFactory,
    ToolCallParserFactory,
    convert_tools_to_definitions,
    convert_tool_call_to_langchain_format,
    ToolCallParseError,
    ToolNotFoundError,
    InvalidToolArgumentsError,
)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool


class MockLLM:
    """Mock LLM client for testing."""
    
    def invoke(self, prompt: str, **kwargs):
        if "tool" in prompt.lower():
            return '{"name": "rag_search", "arguments": {"query": "test"}}'
        return f"Response to: {prompt[:50]}"
    
    def invoke_with_tools(self, prompt: str, tools: list, **kwargs):
        from src.framework.llm.tool_calling import ToolCallResponse, ToolCall
        return ToolCallResponse(
            content="Tool response",
            tool_calls=[
                ToolCall(name="rag_search", arguments={"query": "test"})
            ]
        )
    
    def stream_invoke(self, prompt: str, **kwargs):
        chunks = ["Response", " chunk", " 1", " chunk", " 2"]
        for chunk in chunks:
            yield chunk


class MockTool(BaseTool):
    """Mock tool for testing."""
    
    name: str = "mock_tool"
    description: str = "A mock tool for testing"
    
    def _run(self, query: str) -> str:
        return f"Mock result for: {query}"


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def mock_tool():
    return MockTool()


def test_langchain_adapter_basic(mock_llm):
    """Test basic adapter functionality."""
    adapter = LangChainLLMAdapter(llm_client=mock_llm)
    
    messages = [HumanMessage(content="Hello")]
    result = adapter.invoke(messages)
    
    assert result is not None
    assert hasattr(result, "content")
    assert "Hello" in result.content or "Response" in result.content


def test_langchain_adapter_with_system_message(mock_llm):
    """Test adapter with system message."""
    adapter = LangChainLLMAdapter(llm_client=mock_llm)
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Say hello")
    ]
    result = adapter.invoke(messages)
    
    assert result is not None
    assert hasattr(result, "content")


def test_langchain_adapter_streaming(mock_llm):
    """Test adapter streaming."""
    adapter = LangChainLLMAdapter(llm_client=mock_llm)
    
    messages = [HumanMessage(content="Hello")]
    chunks = list(adapter.stream(messages))
    
    assert len(chunks) > 0
    # BaseChatModel.stream() returns BaseMessageChunk objects (e.g., AIMessageChunk), not ChatGenerationChunk
    # Check that we get message chunk objects with content attribute
    assert all(hasattr(chunk, "content") for chunk in chunks)


def test_langchain_adapter_bind_tools(mock_llm, mock_tool):
    """Test binding tools to adapter."""
    adapter = LangChainLLMAdapter(llm_client=mock_llm)
    bound_adapter = adapter.bind_tools([mock_tool])
    
    assert bound_adapter.bound_tools is not None
    assert len(bound_adapter.bound_tools) == 1
    assert bound_adapter.bound_tools[0].name == "mock_tool"


def test_adapter_factory_create(mock_llm):
    """Test adapter factory."""
    adapter = AdapterFactory.create("openai", mock_llm)
    assert isinstance(adapter, LangChainLLMAdapter)
    
    # Test with different providers
    adapter2 = AdapterFactory.create("gemini", mock_llm)
    assert isinstance(adapter2, LangChainLLMAdapter)


def test_message_serializer_string(mock_tool):
    """Test string message serializer."""
    serializer = MessageSerializerFactory.create_serializer("string")
    
    messages = [
        SystemMessage(content="System"),
        HumanMessage(content="User message")
    ]
    
    serialized = serializer.serialize(messages)
    assert isinstance(serialized, str)
    assert "System" in serialized
    assert "User message" in serialized


def test_message_serializer_json():
    """Test JSON message serializer."""
    serializer = MessageSerializerFactory.create_serializer("json")
    
    messages = [
        SystemMessage(content="System"),
        HumanMessage(content="User message")
    ]
    
    serialized = serializer.serialize(messages)
    assert isinstance(serialized, str)
    # Should be valid JSON
    import json
    data = json.loads(serialized)
    assert "messages" in data


def test_message_serializer_with_tools(mock_tool):
    """Test serializer with tool definitions."""
    serializer = MessageSerializerFactory.create_serializer("string")
    
    from src.framework.llm.tool_calling import ToolDefinition
    
    tools = [
        ToolDefinition(name="test_tool", description="Test tool", parameters={})
    ]
    
    messages = [HumanMessage(content="Hello")]
    serialized = serializer.serialize(messages, tools)
    
    assert "test_tool" in serialized
    assert "Test tool" in serialized


def test_tool_call_parser_json():
    """Test JSON tool call parser."""
    parser = ToolCallParserFactory.create_parser("json")
    
    response = '{"name": "rag_search", "arguments": {"query": "test", "top_k": 4}}'
    tool_calls = parser.parse(response, available_tools=["rag_search"])
    
    assert len(tool_calls) > 0
    assert tool_calls[0].name == "rag_search"
    assert tool_calls[0].arguments["query"] == "test"


def test_tool_call_parser_no_tools():
    """Test parser with no tool calls."""
    parser = ToolCallParserFactory.create_parser("json")
    
    response = "This is a regular response without tool calls."
    tool_calls = parser.parse(response, available_tools=["rag_search"])
    
    assert len(tool_calls) == 0


def test_convert_tools_to_definitions(mock_tool):
    """Test converting tools to definitions."""
    definitions = convert_tools_to_definitions([mock_tool])
    
    assert len(definitions) == 1
    assert definitions[0].name == "mock_tool"
    assert definitions[0].description == "A mock tool for testing"


def test_convert_tools_to_definitions_empty():
    """Test converting empty tool list."""
    definitions = convert_tools_to_definitions(None)
    assert len(definitions) == 0
    
    definitions = convert_tools_to_definitions([])
    assert len(definitions) == 0


def test_convert_tool_call_to_langchain_format():
    """Test converting tool call to LangChain format."""
    from src.framework.llm.tool_calling import ToolCall
    
    tool_call = ToolCall(
        name="rag_search",
        arguments={"query": "test"},
        call_id="call_123"
    )
    
    result = convert_tool_call_to_langchain_format(tool_call)
    
    assert result["name"] == "rag_search"
    assert result["args"] == {"query": "test"}
    assert result["id"] == "call_123"


def test_convert_tool_call_from_dict():
    """Test converting tool call from dict."""
    tool_call_dict = {
        "name": "rag_search",
        "arguments": {"query": "test"}
    }
    
    result = convert_tool_call_to_langchain_format(tool_call_dict)
    
    assert result["name"] == "rag_search"
    assert result["args"] == {"query": "test"}
    assert "id" in result


def test_adapter_native_tool_calling(mock_llm, mock_tool):
    """Test adapter with native tool calling."""
    adapter = LangChainLLMAdapter(
        llm_client=mock_llm,
        use_native_tool_calling=True
    )
    bound_adapter = adapter.bind_tools([mock_tool])
    
    messages = [HumanMessage(content="Use the tool")]
    result = bound_adapter.invoke(messages)
    
    # invoke() returns AIMessage directly, not ChatResult
    assert result is not None
    assert hasattr(result, "content")
    # May or may not have tool calls depending on mock LLM
    assert hasattr(result, "tool_calls")


def test_adapter_error_handling():
    """Test adapter error handling."""
    class FailingLLM:
        def invoke(self, prompt: str, **kwargs):
            raise NotImplementedError("Not implemented")
    
    adapter = LangChainLLMAdapter(llm_client=FailingLLM())
    messages = [HumanMessage(content="Hello")]
    
    with pytest.raises(NotImplementedError):
        adapter.invoke(messages)


def test_adapter_serializer_types():
    """Test different serializer types."""
    serializer_types = ["string", "json", "native"]
    
    for serializer_type in serializer_types:
        serializer = MessageSerializerFactory.create_serializer(serializer_type)
        assert serializer is not None
        
        messages = [HumanMessage(content="Test")]
        result = serializer.serialize(messages)
        assert isinstance(result, str)


def test_adapter_parser_types():
    """Test different parser types."""
    parser_types = ["json", "structured", "native"]
    
    for parser_type in parser_types:
        parser = ToolCallParserFactory.create_parser(parser_type)
        assert parser is not None
        
        # Test parsing (may return empty for native)
        result = parser.parse("test response", available_tools=[])
        assert isinstance(result, list)
