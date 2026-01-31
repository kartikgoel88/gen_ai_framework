"""Tests for framework LLM base (parse_json_from_response, chat default)."""

import pytest

from src.framework.llm.base import LLMClient


class ConcreteLLM(LLMClient):
    def invoke(self, prompt: str, **kwargs):
        return "ok"

    def invoke_structured(self, prompt: str, **kwargs):
        return {"answer": "ok"}


def test_parse_json_from_response_plain_json():
    """parse_json_from_response extracts JSON object from plain text."""
    text = '{"key": "value", "n": 42}'
    out = LLMClient.parse_json_from_response(text)
    assert out == {"key": "value", "n": 42}


def test_parse_json_from_response_markdown_code_block():
    """parse_json_from_response extracts JSON from ```json ... ``` block."""
    text = 'Some text\n```json\n{"a": 1}\n```\nMore text'
    out = LLMClient.parse_json_from_response(text)
    assert out == {"a": 1}


def test_parse_json_from_response_code_block_no_lang():
    """parse_json_from_response handles ```\n{...}\n```."""
    text = '```\n{"x": "y"}\n```'
    out = LLMClient.parse_json_from_response(text)
    assert out == {"x": "y"}


def test_parse_json_from_response_no_json_returns_raw():
    """parse_json_from_response returns {"raw": text} when no JSON found."""
    text = "No json here"
    out = LLMClient.parse_json_from_response(text)
    assert "raw" in out
    assert out["raw"] == "No json here"


def test_parse_json_from_response_empty_string():
    """parse_json_from_response handles empty string."""
    out = LLMClient.parse_json_from_response("")
    assert "raw" in out


def test_chat_default_implementation():
    """LLMClient.chat builds prompt from messages and calls invoke."""
    calls = []

    class RecordingLLM(LLMClient):
        def invoke(self, prompt: str, **kwargs):
            calls.append(prompt)
            return "reply"

        def invoke_structured(self, prompt: str, **kwargs):
            return {}

    llm = RecordingLLM()
    result = llm.chat([
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Bye"},
    ])
    assert result == "reply"
    assert len(calls) == 1
    assert "Hello" in calls[0]
    assert "Bye" in calls[0]


def test_stream_invoke_raises_by_default():
    """LLMClient.stream_invoke raises NotImplementedError by default."""
    llm = ConcreteLLM()
    with pytest.raises(NotImplementedError):
        next(llm.stream_invoke("prompt"))
