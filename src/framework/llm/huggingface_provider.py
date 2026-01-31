"""Hugging Face LLM provider (e.g. Qwen) via Inference API."""

from typing import Any

from .base import LLMClient


class HuggingFaceLLMProvider(LLMClient):
    """LLM client using Hugging Face Inference API (e.g. Qwen from Hugging Face)."""

    def __init__(
        self,
        api_key: str,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.7,
    ):
        self._model_id = model
        self._temperature = temperature
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from langchain_community.llms import HuggingFaceEndpoint
            except ImportError:
                try:
                    from langchain_huggingface import HuggingFaceEndpoint
                except ImportError:
                    raise ImportError(
                        "langchain-community or langchain-huggingface is required for Hugging Face. "
                        "Install with: pip install langchain-community"
                    ) from None
            self._client = HuggingFaceEndpoint(
                repo_id=self._model_id,
                huggingfacehub_api_token=self._api_key,
                temperature=self._temperature,
                max_new_tokens=2048,
            )
        return self._client

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        client = self._get_client()
        return client.invoke(prompt)

    def invoke_structured(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        instruction = (
            "Respond with a single JSON object only. No markdown, no code fences, no extra text."
        )
        full_prompt = f"{prompt}\n\n{instruction}"
        text = self.invoke(full_prompt, **kwargs)
        return self.parse_json_from_response(text)
