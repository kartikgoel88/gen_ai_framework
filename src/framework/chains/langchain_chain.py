"""Wrap LangChain Runnable or LCEL chain so it conforms to the framework Chain interface."""

from typing import Any

from .base import Chain


class LangChainChain(Chain):
    """
    Wraps any LangChain Runnable (e.g. LCEL pipeline) so it can be used as a framework Chain.
    invoke(inputs) delegates to runnable.invoke(inputs).
    """

    def __init__(self, runnable: Any):
        """
        Args:
            runnable: A LangChain Runnable (e.g. PromptTemplate | llm | StrOutputParser(),
                      or RetrievalQA chain). Must have .invoke(input) -> output.
        """
        self._runnable = runnable

    def invoke(self, inputs: dict[str, Any], **kwargs: Any) -> Any:
        """Run the LangChain runnable with the given inputs."""
        return self._runnable.invoke(inputs, **kwargs)
