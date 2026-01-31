"""Classification chain: text -> LLM -> single label from fixed options."""

from typing import Any

from ..llm.base import LLMClient

from .base import Chain

DEFAULT_CLASSIFY_PROMPT = """Classify the following text into exactly one of these categories: {labels}

Text:
{text}

Respond with only the category label, nothing else."""


class ClassificationChain(Chain):
    """Chain that classifies input text into one of the given labels."""

    def __init__(
        self,
        llm: LLMClient,
        labels: list[str],
        prompt_template: str = DEFAULT_CLASSIFY_PROMPT,
    ):
        self._llm = llm
        self._labels = labels
        self._template = prompt_template

    def invoke(self, inputs: dict[str, Any], **kwargs: Any) -> str:
        text = inputs.get("text", inputs.get("content", ""))
        labels_str = ", ".join(self._labels)
        prompt = self._template.format(text=text, labels=labels_str)
        raw = self._llm.invoke(prompt, **kwargs).strip()
        for lab in self._labels:
            if lab.lower() in raw.lower():
                return lab
        return raw
