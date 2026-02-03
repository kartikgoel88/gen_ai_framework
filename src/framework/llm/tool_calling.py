"""Tool calling support for LLM clients."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ToolDefinition:
    """Definition of a tool that can be called by the LLM."""
    
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON schema for parameters
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


@dataclass
class ToolCall:
    """Represents a tool call request from the LLM."""
    
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "name": self.name,
            "arguments": self.arguments
        }
        if self.call_id:
            result["id"] = self.call_id
        return result


@dataclass
class ToolCallResponse:
    """Response from LLM that may contain tool calls."""
    
    content: str
    tool_calls: List[ToolCall]
    
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


class ToolCallingCapability(ABC):
    """Abstract interface for LLM providers that support native tool calling."""
    
    @abstractmethod
    def invoke_with_tools(
        self,
        prompt: str,
        tools: List[ToolDefinition],
        **kwargs: Any
    ) -> ToolCallResponse:
        """
        Invoke LLM with tool definitions and return response with potential tool calls.
        
        Args:
            prompt: The user prompt
            tools: List of available tools
            **kwargs: Additional provider-specific parameters
            
        Returns:
            ToolCallResponse containing content and any tool calls
        """
        ...
