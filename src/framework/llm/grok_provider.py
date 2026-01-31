"""Grok (xAI) LLM provider — OpenAI-compatible API."""

from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .base import LLMClient

XAI_BASE_URL = "https://api.x.ai/v1"


class GrokLLMProvider(LLMClient):
    """LLM client using xAI Grok API (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: str,
        model: str = "grok-2",
        temperature: float = 0.7,
    ):
        self._client = ChatOpenAI(
            base_url=XAI_BASE_URL,
            model=model,
            temperature=temperature,
            openai_api_key=api_key,
        )

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        response = self._client.invoke([HumanMessage(content=prompt)])
        return response.content if hasattr(response, "content") else str(response)

    def invoke_structured(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        instruction = (
            "Respond with a single JSON object only. No markdown, no code fences, no extra text."
        )
        full_prompt = f"{prompt}\n\n{instruction}"
        text = self.invoke(full_prompt, **kwargs)
        return self.parse_json_from_response(text)
