"""Abstract LLM client interface."""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Iterator, Optional


class LLMClient(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt and return the model response text."""
        ...

    @abstractmethod
    def invoke_structured(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Send a prompt and return parsed structured data (e.g. JSON)."""
        ...

    def stream_invoke(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream response tokens. Override in providers that support streaming. Default: raise NotImplementedError."""
        raise NotImplementedError("Streaming not supported by this provider")

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Chat with a list of messages. Default implementation uses a single combined prompt."""
        prompt = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
        )
        return self.invoke(prompt, **kwargs)

    @staticmethod
    def parse_json_from_response(text: str) -> dict[str, Any]:
        """Extract a single JSON object from model response text. Used by invoke_structured implementations."""
        text = (text or "").strip()
        if text.startswith("```"):
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return {"raw": text}
