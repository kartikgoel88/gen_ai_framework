"""Tests for framework prompts (store, templates)."""

import pytest
from pydantic import BaseModel

from src.framework.prompts.store import PromptStore, PromptVersion
from src.framework.prompts.templates import TemplateRunner


class SamplePromptInput(BaseModel):
    question: str
    context: str = ""


def test_prompt_version_default_metadata():
    """PromptVersion initializes metadata to {} if None."""
    pv = PromptVersion(name="q", version="v1", body="Hello")
    assert pv.metadata == {}


def test_prompt_store_put_get(tmp_path):
    """PromptStore put and get round-trip."""
    store = PromptStore(base_path=str(tmp_path))
    store.put("greet", "v1", "Hello {name}!")
    pv = store.get("greet", "v1")
    assert pv is not None
    assert pv.name == "greet"
    assert pv.version == "v1"
    assert pv.body == "Hello {name}!"


def test_prompt_store_get_missing_returns_none(tmp_path):
    """PromptStore get returns None when prompt does not exist."""
    store = PromptStore(base_path=str(tmp_path))
    assert store.get("missing", "v1") is None


def test_prompt_store_list_versions(tmp_path):
    """PromptStore list_versions returns sorted version tags."""
    store = PromptStore(base_path=str(tmp_path))
    store.put("p", "v2", "body2")
    store.put("p", "v1", "body1")
    store.put("p", "v3", "body3")
    versions = store.list_versions("p")
    assert versions == ["v1", "v2", "v3"]


def test_prompt_store_list_names(tmp_path):
    """PromptStore list_names returns unique prompt names."""
    store = PromptStore(base_path=str(tmp_path))
    store.put("a", "v1", "a")
    store.put("b", "v1", "b")
    store.put("a", "v2", "a2")
    names = store.list_names()
    assert set(names) == {"a", "b"}


def test_template_runner_extracts_variables():
    """TemplateRunner extracts {variable} placeholders from template."""
    llm = type("MockLLM", (), {"invoke": lambda self, prompt, **kw: prompt, "invoke_structured": lambda self, prompt, **kw: {}})()
    runner = TemplateRunner(llm=llm, template="Q: {question}\nC: {context}")
    assert "question" in runner._input_variables
    assert "context" in runner._input_variables


def test_template_runner_run_formats_and_invokes():
    """TemplateRunner run formats template and calls LLM."""
    out = []

    class MockLLM:
        def invoke(self, prompt, **kwargs):
            out.append(prompt)
            return "answer"

        def invoke_structured(self, prompt, **kwargs):
            return {"answer": "structured"}

    runner = TemplateRunner(llm=MockLLM(), template="Question: {question}")
    result = runner.run({"question": "What is 2+2?"})
    assert result == "answer"
    assert len(out) == 1
    assert "What is 2+2?" in out[0]


def test_template_runner_run_with_pydantic_model():
    """TemplateRunner run accepts Pydantic model and formats template."""
    class MockLLM:
        def invoke(self, prompt, **kwargs):
            return prompt

    runner = TemplateRunner(llm=MockLLM(), template="{question} | {context}", input_model=SamplePromptInput)
    result = runner.run(SamplePromptInput(question="Q", context="C"))
    assert "Q" in result and "C" in result
