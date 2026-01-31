"""Prompt chain: template + LLM → formatted prompt → response text."""

from typing import Any

from ..llm.base import LLMClient

from .base import Chain


class PromptChain(Chain):
    """Chain that formats a prompt template with inputs and invokes the LLM."""

    def __init__(
        self,
        llm: LLMClient,
        template: str,
        input_variables: list[str] | None = None,
    ):
        """
        Args:
            llm: LLM client (invoke).
            template: Prompt string with {variable} placeholders.
            input_variables: Names of expected keys in inputs; if None, inferred from {x} in template.
        """
        self._llm = llm
        self._template = template
        if input_variables is not None:
            self._input_variables = input_variables
        else:
            import re
            self._input_variables = list(re.findall(r"\{(\w+)\}", template))

    def invoke(self, inputs: dict[str, Any], **kwargs: Any) -> str:
        """Format template with inputs and return LLM response."""
        try:
            formatted = self._template.format(**{k: inputs.get(k, "") for k in self._input_variables})
        except KeyError as e:
            raise ValueError(f"Missing input for chain: {e}") from e
        return self._llm.invoke(formatted, **kwargs)
