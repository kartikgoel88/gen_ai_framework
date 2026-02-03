"""Chains: compose prompt, RAG, LLM, and output parsing.

This module provides chain implementations for composing LLM operations:
- Prompt chains: Simple prompt → LLM → output
- RAG chains: Retrieve → LLM → answer
- Structured chains: Extract structured data
- Specialized chains: Summarization, classification, extraction
- Pipeline: Multi-step chain composition
- LangChain integration: Wrappers for LangChain LCEL chains
"""

from .base import Chain
from .prompt_chain import PromptChain
from .rag_chain import RAGChain
from .structured_chain import StructuredChain
from .summarization_chain import SummarizationChain
from .classification_chain import ClassificationChain
from .extraction_chain import ExtractionChain
from .pipeline import Pipeline, PipelineStep, pipeline_from_config
from .langchain_chain import LangChainChain
from .langchain_factory import (
    build_langchain_prompt_chain,
    build_langchain_chat_prompt_chain,
    build_langchain_rag_chain,
)

__all__ = [
    # Base
    "Chain",
    # Chain implementations
    "PromptChain",
    "RAGChain",
    "StructuredChain",
    "SummarizationChain",
    "ClassificationChain",
    "ExtractionChain",
    # Pipeline
    "Pipeline",
    "PipelineStep",
    "pipeline_from_config",
    # LangChain integration
    "LangChainChain",
    "build_langchain_prompt_chain",
    "build_langchain_chat_prompt_chain",
    "build_langchain_rag_chain",
]

# Note: LangChainLLMAdapter is now in framework.adapters, not chains
# Import it from there if needed: from ..adapters import LangChainLLMAdapter
