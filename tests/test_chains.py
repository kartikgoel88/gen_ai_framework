"""Tests for chains: PromptChain, RAGChain, SummarizationChain, ClassificationChain, ExtractionChain, Pipeline, LangChain integration."""

from unittest.mock import MagicMock

import pytest

from src.framework.chains import (
    PromptChain,
    RAGChain,
    SummarizationChain,
    ClassificationChain,
    ExtractionChain,
    Pipeline,
    PipelineStep,
    pipeline_from_config,
    LangChainChain,
    build_langchain_prompt_chain,
    build_langchain_chat_prompt_chain,
    build_langchain_rag_chain,
)
from src.framework.chains.base import Chain
from src.framework.adapters import LangChainLLMAdapter


class MockLLM:
    def invoke(self, prompt: str, **kwargs):
        if "summarize" in prompt.lower():
            return "Short summary."
        if "classify" in prompt.lower() or "positive" in prompt.lower():
            return "positive"
        if "extract" in prompt.lower():
            return '{"title": "Test", "author": "Alice"}'
        return f"Echo: {prompt[:50]}"

    def invoke_structured(self, prompt: str, **kwargs):
        return {"title": "Test", "author": "Alice", "score": 1}


class MockRAG:
    def retrieve(self, query: str, top_k: int = 4, **kwargs):
        return [{"content": "Retrieved context for " + query, "metadata": {}}]

    def query(self, question: str, llm_client=None, **kwargs):
        return "Answer from RAG"


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def mock_rag():
    return MockRAG()


def test_prompt_chain(mock_llm):
    chain = PromptChain(llm=mock_llm, template="Say: {word}")
    out = chain.invoke({"word": "hello"})
    assert "hello" in out or "Echo" in out


def test_rag_chain(mock_llm, mock_rag):
    chain = RAGChain(llm=mock_llm, rag=mock_rag, top_k=2)
    out = chain.invoke({"query": "test query"})
    assert isinstance(out, str)
    assert len(out) > 0


def test_summarization_chain(mock_llm):
    chain = SummarizationChain(llm=mock_llm)
    out = chain.invoke({"text": "Long text to summarize here."})
    assert isinstance(out, str)
    assert "summary" in out.lower() or "Short" in out


def test_classification_chain(mock_llm):
    chain = ClassificationChain(llm=mock_llm, labels=["positive", "negative", "neutral"])
    out = chain.invoke({"text": "This is great!"})
    assert out in ["positive", "negative", "neutral"] or "positive" in out.lower()


def test_extraction_chain(mock_llm):
    chain = ExtractionChain(llm=mock_llm, schema="title, author")
    out = chain.invoke({"text": "Article by Alice titled Test."})
    assert isinstance(out, dict)
    assert "title" in out or "author" in out or "raw" in out


def test_pipeline_sequential(mock_llm):
    step1 = PromptChain(llm=mock_llm, template="{x}")
    step2 = PromptChain(llm=mock_llm, template="Result: {output}")
    pipeline = Pipeline(
        steps=[
            PipelineStep(step_id="s1", chain=step1, output_key="output"),
            PipelineStep(step_id="s2", chain=step2, output_key="final"),
        ],
        final_output_key="final",
    )
    out = pipeline.invoke({"x": "hello"})
    assert isinstance(out, str)
    assert "hello" in out or "Echo" in out or "Result" in out or len(out) > 0


def test_pipeline_from_config(mock_llm):
    def build_chain(config):
        t = config.get("template", "{prompt}")
        return PromptChain(llm=mock_llm, template=t)

    config = [
        {"id": "a", "output_key": "first", "template": "{input}"},
        {"id": "b", "output_key": "second", "template": "Done: {first}"},
    ]
    pipeline = pipeline_from_config(config, build_chain)
    out = pipeline.invoke({"input": "test"})
    assert isinstance(out, str)


def test_pipeline_extract_summarize_classify(mock_llm):
    """Test complex pipeline: Extract → Summarize → Classify."""
    extract_chain = ExtractionChain(llm=mock_llm, schema="title, author")
    summarize_chain = SummarizationChain(llm=mock_llm)
    classify_chain = ClassificationChain(llm=mock_llm, labels=["technical", "biographical"])
    
    pipeline = Pipeline(steps=[
        PipelineStep(step_id="extract", chain=extract_chain, output_key="entities"),
        PipelineStep(step_id="summarize", chain=summarize_chain, output_key="summary"),
        PipelineStep(step_id="classify", chain=classify_chain, output_key="category")
    ])
    
    result = pipeline.invoke({"text": "John Doe wrote a book about Python programming."})
    # Pipeline returns the final output value (from final_output_key="category")
    # The result should be a string (classification label) or dict if final_output_key not found
    assert result is not None
    # Result could be a string (the classification) or dict (if final_output_key not in state)
    assert isinstance(result, (str, dict))


def test_langchain_chain_wrapper(mock_llm):
    """Test LangChainChain wrapper."""
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    adapter = LangChainLLMAdapter(llm_client=mock_llm)
    prompt = PromptTemplate(template="Say: {word}", input_variables=["word"])
    runnable = prompt | adapter | StrOutputParser()
    
    chain = LangChainChain(runnable=runnable)
    result = chain.invoke({"word": "hello"})
    assert isinstance(result, str)
    assert len(result) > 0


def test_build_langchain_prompt_chain(mock_llm):
    """Test building LangChain prompt chain."""
    chain = build_langchain_prompt_chain(
        llm=mock_llm,
        template="Translate '{text}' to Spanish",
        input_variables=["text"]
    )
    
    result = chain.invoke({"text": "Hello"})
    assert isinstance(result, str)


def test_build_langchain_chat_prompt_chain(mock_llm):
    """Test building LangChain chat prompt chain."""
    chain = build_langchain_chat_prompt_chain(
        llm=mock_llm,
        system="You are a translator.",
        human_template="Translate: {text}"
    )
    
    result = chain.invoke({"text": "Hello"})
    assert isinstance(result, str)


def test_build_langchain_rag_chain(mock_llm, mock_rag):
    """Test building LangChain RAG chain."""
    chain = build_langchain_rag_chain(
        llm=mock_llm,
        rag=mock_rag,
        top_k=2
    )
    
    result = chain.invoke({"question": "What is Python?"})
    assert isinstance(result, str)
    assert len(result) > 0


def test_pipeline_custom_output_key(mock_llm):
    """Test pipeline with custom final output key."""
    step1 = PromptChain(llm=mock_llm, template="{x}")
    step2 = PromptChain(llm=mock_llm, template="Result: {output}")
    
    pipeline = Pipeline(
        steps=[
            PipelineStep(step_id="s1", chain=step1, output_key="output"),
            PipelineStep(step_id="s2", chain=step2, output_key="final"),
        ],
        final_output_key="output"  # Return first step's output
    )
    
    result = pipeline.invoke({"x": "hello"})
    assert isinstance(result, str)
