"""Google Gemini LLM provider."""

from typing import Any

from langchain_core.messages import HumanMessage

from .base import LLMClient


class GeminiLLMProvider(LLMClient):
    """LLM client using Google Gemini API (langchain-google-genai)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.7,
    ):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                "langchain-google-genai is required for Gemini. Install with: pip install langchain-google-genai"
            ) from e
        self._client = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key,
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
