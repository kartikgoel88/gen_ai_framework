"""LLM call tracing: log prompt, response, latency, and optional token usage."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..llm.base import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class TraceEntry:
    """Single LLM call trace."""

    operation: str  # "invoke" | "invoke_structured" | "chat"
    prompt: str
    response: Any  # str or dict for structured
    latency_seconds: float
    prompt_length: int = 0
    response_length: int = 0
    token_usage: Optional[dict[str, int]] = None  # input_tokens, output_tokens, total_tokens
    metadata: dict[str, Any] = field(default_factory=dict)


class TracingLLMClient(LLMClient):
    """Wraps an LLMClient and logs each call (prompt, response, latency)."""

    def __init__(
        self,
        inner: LLMClient,
        log_level: int = logging.INFO,
        callback: Optional[Callable[[TraceEntry], None]] = None,
    ):
        self._inner = inner
        self._log_level = log_level
        self._callback = callback

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        start = time.perf_counter()
        response = ""
        try:
            response = self._inner.invoke(prompt, **kwargs)
            return response
        finally:
            elapsed = time.perf_counter() - start
            entry = TraceEntry(
                operation="invoke",
                prompt=prompt,
                response=response,
                latency_seconds=elapsed,
                prompt_length=len(prompt),
                response_length=len(response),
                metadata=kwargs,
            )
            self._emit(entry)

    def invoke_structured(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        start = time.perf_counter()
        response: dict[str, Any] = {}
        try:
            response = self._inner.invoke_structured(prompt, **kwargs)
            return response
        finally:
            elapsed = time.perf_counter() - start
            entry = TraceEntry(
                operation="invoke_structured",
                prompt=prompt,
                response=response,
                latency_seconds=elapsed,
                prompt_length=len(prompt),
                response_length=len(str(response)),
                metadata=kwargs,
            )
            self._emit(entry)

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        start = time.perf_counter()
        response = ""
        try:
            response = self._inner.chat(messages, **kwargs)
            return response
        finally:
            elapsed = time.perf_counter() - start
            prompt_repr = str(messages)
            entry = TraceEntry(
                operation="chat",
                prompt=prompt_repr,
                response=response,
                latency_seconds=elapsed,
                prompt_length=len(prompt_repr),
                response_length=len(response),
                metadata=kwargs,
            )
            self._emit(entry)

    def _emit(self, entry: TraceEntry) -> None:
        logger.log(
            self._log_level,
            "LLM trace | op=%s latency=%.3fs prompt_len=%s response_len=%s",
            entry.operation,
            entry.latency_seconds,
            entry.prompt_length,
            entry.response_length,
        )
        if self._callback:
            try:
                self._callback(entry)
            except Exception as e:
                logger.warning("Tracing callback failed: %s", e)
