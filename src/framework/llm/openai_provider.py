"""OpenAI LLM provider implementation."""

from typing import Any, Iterator, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .base import LLMClient


class OpenAILLMProvider(LLMClient):
    """LLM client using OpenAI API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
    ):
        self._client = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=api_key,
        )

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        response = self._client.invoke([HumanMessage(content=prompt)])
        return response.content if hasattr(response, "content") else str(response)

    def stream_invoke(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream response chunks (token-by-token)."""
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
