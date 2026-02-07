"""Abstract LLM client interface."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Iterator, Optional

from ..utils.json_utils import parse_json_from_response as _parse_json_from_response
from .tool_calling import ToolCallResponse, ToolDefinition


class LLMClient(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt and return the model response text."""
        ...

    @abstractmethod
    def invoke_structured(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Send a prompt and return parsed structured data (e.g. JSON)."""
        ...

    def stream_invoke(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream response tokens. Override in providers that support streaming. Default: raise NotImplementedError."""
        raise NotImplementedError("Streaming not supported by this provider")

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Chat with a list of messages. Default implementation uses a single combined prompt."""
        prompt = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
        )
        return self.invoke(prompt, **kwargs)

    def invoke_with_tools(
        self,
        prompt: str,
        tools: list[ToolDefinition],
        **kwargs: Any
    ) -> ToolCallResponse:
        """
        Invoke LLM with tool definitions. Default implementation uses structured output.
        
        Override in providers that support native tool calling (e.g., OpenAI function calling).
        
        Args:
            prompt: The user prompt
            tools: List of available tools
            **kwargs: Additional parameters
            
        Returns:
            ToolCallResponse containing content and any tool calls
        """
        # Default implementation: use structured output to request tool calls
        tool_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": [t.name for t in tools]},
                "arguments": {"type": "object"}
            },
            "required": ["name", "arguments"]
        }
        
        structured_prompt = f"{prompt}\n\nAvailable tools:\n"
        for tool in tools:
            structured_prompt += f"- {tool.name}: {tool.description}\n"
        structured_prompt += "\nIf you need to use a tool, respond with JSON: {\"name\": \"tool_name\", \"arguments\": {...}}"
        
        try:
            result = self.invoke_structured(structured_prompt, **kwargs)
            from .tool_calling import ToolCall
            
            tool_calls = []
            if isinstance(result, dict) and "name" in result:
                tool_calls.append(ToolCall(
                    name=result.get("name", ""),
                    arguments=result.get("arguments", {})
                ))
            
            return ToolCallResponse(
                content=result.get("raw", "") if isinstance(result, dict) else "",
                tool_calls=tool_calls
            )
        except Exception:
            # Fallback to regular invoke
            content = self.invoke(prompt, **kwargs)
            return ToolCallResponse(content=content, tool_calls=[])

    @staticmethod
    def parse_json_from_response(text: str) -> dict[str, Any]:
        """Extract a single JSON object from model response text. Used by invoke_structured implementations."""
        return _parse_json_from_response(text, default_raw=True)
