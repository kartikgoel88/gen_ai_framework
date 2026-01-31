"""Summarization chain: long text -> LLM -> summary."""

from typing import Any

from ..llm.base import LLMClient

from .base import Chain

DEFAULT_SUMMARY_PROMPT = """Summarize the following text concisely. Preserve key facts and conclusions.

Text:
{text}

Summary:"""


class SummarizationChain(Chain):
    """Chain that summarizes input text using the LLM."""

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = DEFAULT_SUMMARY_PROMPT,
        max_chars: int = 0,
    ):
        self._llm = llm
        self._template = prompt_template
        self._max_chars = max_chars

    def invoke(self, inputs: dict[str, Any], **kwargs: Any) -> str:
        text = inputs.get("text", inputs.get("content", ""))
        if self._max_chars > 0 and len(text) > self._max_chars:
            text = text[: self._max_chars] + "..."
        prompt = self._template.format(text=text)
        return self._llm.invoke(prompt, **kwargs)
