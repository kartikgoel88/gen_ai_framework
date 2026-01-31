"""Abstract chain interface: compose prompt, retrieval, LLM, and output parsing."""

from abc import ABC, abstractmethod
from typing import Any


class Chain(ABC):
    """Abstract interface for chains: invoke with inputs dict, return output."""

    @abstractmethod
    def invoke(self, inputs: dict[str, Any], **kwargs: Any) -> Any:
        """Run the chain with the given inputs. Returns chain output (str, dict, or parsed object)."""
        ...

    def __call__(self, inputs: dict[str, Any], **kwargs: Any) -> Any:
        """Convenience: chain(inputs) == chain.invoke(inputs)."""
        return self.invoke(inputs, **kwargs)
