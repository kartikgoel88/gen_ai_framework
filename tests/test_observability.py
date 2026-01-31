"""Tests for framework observability (tracing)."""

from unittest.mock import MagicMock

import pytest

from src.framework.observability.tracing import TraceEntry, TracingLLMClient
from src.framework.llm.base import LLMClient


class MockLLM(LLMClient):
    def invoke(self, prompt: str, **kwargs):
        return f"response:{prompt[:10]}"

    def invoke_structured(self, prompt: str, **kwargs):
        return {"answer": "structured", "prompt_preview": prompt[:10]}


def test_trace_entry_dataclass():
    """TraceEntry holds operation, prompt, response, latency."""
    entry = TraceEntry(
        operation="invoke",
        prompt="hello",
        response="world",
        latency_seconds=0.5,
        prompt_length=5,
        response_length=5,
    )
    assert entry.operation == "invoke"
    assert entry.prompt == "hello"
    assert entry.response == "world"
    assert entry.latency_seconds == 0.5
    assert entry.metadata == {}


def test_tracing_llm_client_invoke_delegates_and_captures():
    """TracingLLMClient invoke calls inner LLM and captures trace."""
    inner = MockLLM()
    traces = []

    def capture(e: TraceEntry):
        traces.append(e)

    client = TracingLLMClient(inner=inner, callback=capture)
    result = client.invoke("test prompt")
    assert "response:" in result
    assert result == inner.invoke("test prompt")
    assert len(traces) == 1
    assert traces[0].operation == "invoke"
    assert traces[0].prompt == "test prompt"
    assert traces[0].response == result
    assert traces[0].latency_seconds >= 0


def test_tracing_llm_client_invoke_structured():
    """TracingLLMClient invoke_structured delegates and captures."""
    inner = MockLLM()
    traces = []
    client = TracingLLMClient(inner=inner, callback=traces.append)
    out = client.invoke_structured("structured prompt")
    assert out.get("answer") == "structured"
    assert len(traces) == 1
    assert traces[0].operation == "invoke_structured"
    assert traces[0].response == out


def test_tracing_llm_client_chat():
    """TracingLLMClient chat delegates to inner and captures."""
    inner = MockLLM()
    traces = []
    client = TracingLLMClient(inner=inner, callback=traces.append)
    messages = [{"role": "user", "content": "Hi"}]
    result = client.chat(messages)
    assert "response:" in result
    assert len(traces) == 1
    assert traces[0].operation == "chat"
