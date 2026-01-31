"""Abstract agent interface."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class AgentBase(ABC):
    """Interface for agents: accept message, optional tools, return response."""

    @abstractmethod
    def invoke(
        self,
        message: str,
        *,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Run the agent and return the final response text."""
        ...

    def get_tools_description(self) -> list[dict[str, Any]]:
        """Return list of {name, description} for tools available to this agent."""
        return []
