"""Observability: LLM tracing and evaluation harness."""

from .tracing import TracingLLMClient, TraceEntry
from .eval import EvalHarness, EvalResult, EvalDatasetItem, evaluate_multiple_models, ModelSpec

__all__ = [
    "TracingLLMClient",
    "TraceEntry",
    "EvalHarness",
    "EvalResult",
    "EvalDatasetItem",
    "evaluate_multiple_models",
    "ModelSpec",
]
