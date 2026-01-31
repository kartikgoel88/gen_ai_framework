"""Tests for LangGraph: RAG graph invoke."""

from unittest.mock import MagicMock

import pytest

from src.framework.graph.rag_graph import build_rag_graph, RAGGraphState


class MockLLM:
    def invoke(self, prompt: str, **kwargs):
        return "Generated answer from context."


class MockRAG:
    def retrieve(self, query: str, top_k: int = 4, **kwargs):
        return [
            {"content": "Context chunk 1 for " + query, "metadata": {}},
            {"content": "Context chunk 2", "metadata": {}},
        ]


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def mock_rag():
    return MockRAG()


def test_rag_graph_state_typing():
    state: RAGGraphState = {"query": "q", "context": "", "response": ""}
    assert state["query"] == "q"


def test_build_rag_graph_invoke(mock_llm, mock_rag):
    graph = build_rag_graph(llm=mock_llm, rag=mock_rag, top_k=2)
    result = graph.invoke({"query": "What is the policy?"})
    assert "query" in result
    assert "context" in result
    assert "response" in result
    assert result["query"] == "What is the policy?"
    assert len(result["context"]) > 0
    assert result["response"] == "Generated answer from context."
