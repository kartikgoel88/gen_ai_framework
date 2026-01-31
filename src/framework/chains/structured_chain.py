"""Structured chain: template + LLM invoke_structured → parsed dict (e.g. JSON)."""

from typing import Any, TypeVar

from ..llm.base import LLMClient

from .base import Chain

T = TypeVar("T")


class StructuredChain(Chain):
    """Chain that formats a prompt, calls invoke_structured, and returns a dict (or parsed model)."""

    def __init__(
        self,
        llm: LLMClient,
        template: str,
        input_variables: list[str] | None = None,
        instruction: str = "Respond with a single JSON object only. No markdown, no code fences.",
    ):
        """
        Args:
            llm: LLM client (invoke_structured).
            template: Prompt template with {variable} placeholders.
            input_variables: Expected input keys; if None, inferred from template.
            instruction: Appended to prompt to encourage JSON output.
        """
        self._llm = llm
        self._template = template
        self._instruction = instruction
        if input_variables is not None:
            self._input_variables = input_variables
        else:
            import re
            self._input_variables = list(re.findall(r"\{(\w+)\}", template))

    def invoke(self, inputs: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Format template, call invoke_structured, return parsed dict."""
        try:
            formatted = self._template.format(**{k: inputs.get(k, "") for k in self._input_variables})
        except KeyError as e:
            raise ValueError(f"Missing input for chain: {e}") from e
        full_prompt = f"{formatted}\n\n{self._instruction}"
        return self._llm.invoke_structured(full_prompt, **kwargs)
