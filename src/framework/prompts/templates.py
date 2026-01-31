"""Templates: variables and validation (e.g. Pydantic) before calling LLM."""

from typing import Any, Optional, Type

from pydantic import BaseModel

from ..llm.base import LLMClient


class TemplateRunner:
    """Run a prompt template with variable validation. Supports Pydantic model for inputs."""

    def __init__(
        self,
        llm: LLMClient,
        template: str,
        input_model: Optional[Type[BaseModel]] = None,
        input_variables: Optional[list[str]] = None,
    ):
        self._llm = llm
        self._template = template
        self._input_model = input_model
        if input_variables is not None:
            self._input_variables = input_variables
        else:
            import re
            self._input_variables = list(re.findall(r"\{(\w+)\}", template))

    def run(self, inputs: dict[str, Any] | BaseModel, **kwargs: Any) -> str:
        """Validate inputs (if Pydantic model), format template, invoke LLM."""
        if isinstance(inputs, BaseModel):
            data = inputs.model_dump()
        else:
            data = dict(inputs)
        if self._input_model is not None and not isinstance(inputs, self._input_model):
            data = self._input_model(**data).model_dump()
        for k in self._input_variables:
            if k not in data:
                data[k] = ""
        formatted = self._template.format(**data)
        return self._llm.invoke(formatted, **kwargs)

    def run_structured(self, inputs: dict[str, Any] | BaseModel, **kwargs: Any) -> dict[str, Any]:
        """Validate inputs, format template, invoke LLM with structured (JSON) output."""
        if isinstance(inputs, BaseModel):
            data = inputs.model_dump()
        else:
            data = dict(inputs)
        if self._input_model is not None and not isinstance(inputs, self._input_model):
            data = self._input_model(**data).model_dump()
        for k in self._input_variables:
            if k not in data:
                data[k] = ""
        formatted = self._template.format(**data)
        return self._llm.invoke_structured(formatted, **kwargs)
