"""Local LLM provider — OpenAI-compatible endpoint (Ollama, LM Studio, etc.). PII-safe: data stays on your machine."""

from typing import Any, Iterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .base import LLMClient

# Default for Ollama; use env LLM_LOCAL_BASE_URL to override (e.g. http://localhost:1234/v1 for LM Studio)
DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"


class LocalLLMProvider(LLMClient):
    """LLM client for a local OpenAI-compatible endpoint (Ollama, LM Studio, llama.cpp server).
    No API key required; data never leaves the host — suitable for PII-sensitive workloads."""

    def __init__(
        self,
        base_url: str = DEFAULT_LOCAL_BASE_URL,
        model: str = "llama3.2",
        temperature: float = 0.7,
        api_key: str | None = None,
    ):
        self._client = ChatOpenAI(
            base_url=base_url,
            model=model,
            temperature=temperature,
            openai_api_key=api_key or "local",
        )

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        response = self._client.invoke([HumanMessage(content=prompt)])
        return response.content if hasattr(response, "content") else str(response)

    def stream_invoke(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        for chunk in self._client.stream([HumanMessage(content=prompt)]):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content

    def invoke_structured(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        instruction = (
            "Respond with a single JSON object only. No markdown, no code fences, no extra text."
        )
        full_prompt = f"{prompt}\n\n{instruction}"
        text = self.invoke(full_prompt, **kwargs)
        return self.parse_json_from_response(text)
