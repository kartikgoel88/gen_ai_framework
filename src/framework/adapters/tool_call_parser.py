"""Tool call parsing strategies for different LLM response formats."""

import json
import re
from abc import ABC, abstractmethod
from typing import List, Optional

from ..llm.tool_calling import ToolCall


class ToolCallParser(ABC):
    """Abstract base class for parsing tool calls from LLM responses."""
    
    @abstractmethod
    def parse(self, response: str, available_tools: Optional[List[str]] = None) -> List[ToolCall]:
        """
        Parse tool calls from LLM response.
        
        Args:
            response: The raw response from the LLM
            available_tools: Optional list of valid tool names for validation
            
        Returns:
            List of parsed ToolCall objects
        """
        ...


class JSONToolCallParser(ToolCallParser):
    """Parser for JSON-formatted tool calls (current implementation)."""
    
    def parse(self, response: str, available_tools: Optional[List[str]] = None) -> List[ToolCall]:
        """Parse JSON tool calls from response."""
        tool_calls = []
        response_stripped = response.strip()
        
        # Try to parse entire response as JSON
        if response_stripped.startswith("{"):
            try:
                tool_call_data = json.loads(response_stripped)
                tool_call = self._parse_single_tool_call(tool_call_data, available_tools)
                if tool_call:
                    tool_calls.append(tool_call)
                    return tool_calls
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON objects in text
        json_pattern = r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                tool_call_data = json.loads(match)
                tool_call = self._parse_single_tool_call(tool_call_data, available_tools)
                if tool_call:
                    tool_calls.append(tool_call)
            except (json.JSONDecodeError, KeyError):
                continue
        
        return tool_calls
    
    def _parse_single_tool_call(
        self,
        data: dict,
        available_tools: Optional[List[str]] = None
    ) -> Optional[ToolCall]:
        """Parse a single tool call from dictionary."""
        if not isinstance(data, dict) or "name" not in data:
            return None
        
        tool_name = data.get("name", "")
        
        # Validate tool name if available tools provided
        if available_tools and tool_name not in available_tools:
            return None
        
        tool_args = data.get("arguments") or data.get("args", {})
        call_id = data.get("id") or f"call_{tool_name}_{abs(hash(str(tool_args)))}"
        
        return ToolCall(
            name=tool_name,
            arguments=tool_args,
            call_id=call_id
        )


class StructuredOutputParser(ToolCallParser):
    """Parser for structured output format (JSON schema)."""
    
    def parse(self, response: str, available_tools: Optional[List[str]] = None) -> List[ToolCall]:
        """Parse structured output format."""
        # Similar to JSON parser but validates against schema
        json_parser = JSONToolCallParser()
        tool_calls = json_parser.parse(response, available_tools)
        
        # Additional validation for structured output
        for tool_call in tool_calls:
            if not self._validate_tool_call(tool_call):
                tool_calls.remove(tool_call)
        
        return tool_calls
    
    def _validate_tool_call(self, tool_call: ToolCall) -> bool:
        """Validate tool call structure."""
        if not tool_call.name:
            return False
        if not isinstance(tool_call.arguments, dict):
            return False
        return True


class NativeToolCallParser(ToolCallParser):
    """Parser for native tool calling format (e.g., OpenAI function calling)."""
    
    def parse(self, response: str, available_tools: Optional[List[str]] = None) -> List[ToolCall]:
        """
        Parse native tool calls.
        
        This is a placeholder - native tool calls are typically handled
        by the provider's SDK and returned in a structured format.
        """
        # Native parsers typically work with structured responses from SDKs
        # This would be implemented by provider-specific adapters
        return []


class ToolCallParserFactory:
    """Factory for creating appropriate tool call parsers."""
    
    @staticmethod
    def create_parser(parser_type: str = "json") -> ToolCallParser:
        """
        Create a tool call parser based on type.
        
        Args:
            parser_type: Type of parser ("json", "structured", "native")
            
        Returns:
            ToolCallParser instance
        """
        parsers = {
            "json": JSONToolCallParser,
            "structured": StructuredOutputParser,
            "native": NativeToolCallParser,
        }
        
        parser_class = parsers.get(parser_type.lower(), JSONToolCallParser)
        return parser_class()
