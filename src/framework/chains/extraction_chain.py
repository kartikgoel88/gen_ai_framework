"""Extraction chain: text -> LLM -> structured fields (e.g. entities, key-value)."""

from typing import Any

from ..llm.base import LLMClient

from .base import Chain

DEFAULT_EXTRACTION_PROMPT = """Extract structured information from the following text. Return a JSON object with the requested fields.

Text:
{text}

Requested fields (or schema): {schema}

Return only a valid JSON object, no markdown or explanation."""


class ExtractionChain(Chain):
    """Chain that extracts structured data from text using the LLM (invoke_structured)."""

    def __init__(
        self,
        llm: LLMClient,
        schema: str | list[str],
        prompt_template: str = DEFAULT_EXTRACTION_PROMPT,
    ):
        self._llm = llm
        self._schema = schema if isinstance(schema, str) else ", ".join(schema)
        self._template = prompt_template

    def invoke(self, inputs: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        text = inputs.get("text", inputs.get("content", ""))
        prompt = self._template.format(text=text, schema=self._schema)
        return self._llm.invoke_structured(prompt, **kwargs)
