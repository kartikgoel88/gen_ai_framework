"""Tests for agents: tools list, invoke with mock LLM and tools, adapter integration."""

from unittest.mock import MagicMock

import pytest

from src.framework.agents.base import AgentBase
from src.framework.agents.tools import build_framework_tools, build_rag_tool
from src.framework.adapters import LangChainLLMAdapter


class MockRAG:
    def retrieve(self, query: str, top_k: int = 4, **kwargs):
        return [{"content": "Retrieved: " + query, "metadata": {}}]


class MockMCP:
    def list_tools(self):
        return [
            {"name": "mock_tool", "description": "A mock MCP tool"},
        ]

    def call_tool(self, name: str, arguments: dict):
        return {"result": "mock result"}


@pytest.fixture
def mock_rag():
    return MockRAG()


@pytest.fixture
def mock_mcp():
    return MockMCP()


def test_build_rag_tool(mock_rag):
    tool = build_rag_tool(mock_rag)
    assert tool.name == "rag_search"
    out = tool._run("test query", top_k=2)
    assert "Retrieved" in out or "test query" in out


def test_build_framework_tools(mock_rag, mock_mcp):
    tools = build_framework_tools(rag_client=mock_rag, mcp_client=mock_mcp)
    names = [t.name for t in tools]
    assert "rag_search" in names
    assert len(tools) >= 1


def test_agent_tools_list(mock_rag, mock_mcp):
    tools = build_framework_tools(rag_client=mock_rag, mcp_client=mock_mcp)
    descriptions = [{"name": t.name, "description": t.description} for t in tools]
    assert any(d["name"] == "rag_search" for d in descriptions)


def test_agent_invoke_requires_langchain():
    """Agent invoke uses LangChain AgentExecutor; we only test that tools are built correctly."""
    mock_rag = MockRAG()
    mock_mcp = MockMCP()
    tools = build_framework_tools(rag_client=mock_rag, mcp_client=mock_mcp)
    assert len(tools) >= 1
    rag_tool = next(t for t in tools if t.name == "rag_search")
    result = rag_tool.invoke({"query": "hello", "top_k": 2})
    assert "hello" in result or "Retrieved" in result


def test_adapter_with_agent_tools(mock_rag):
    """Test adapter integration with agent tools."""
    class MockLLM:
        def invoke(self, prompt: str, **kwargs):
            return "Mock response"
    
    llm = MockLLM()
    adapter = LangChainLLMAdapter(llm_client=llm)
    
    tools = build_framework_tools(rag_client=mock_rag, mcp_client=None, enable_web_search=False)
    bound_adapter = adapter.bind_tools(tools)
    
    assert bound_adapter.bound_tools is not None
    assert len(bound_adapter.bound_tools) > 0
    assert any(t.name == "rag_search" for t in bound_adapter.bound_tools)
