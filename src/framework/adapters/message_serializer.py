"""Message serialization strategies for different LLM providers."""

import json
from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage

from ..llm.tool_calling import ToolDefinition


class MessageSerializer(ABC):
    """Abstract base class for serializing LangChain messages to provider format."""
    
    @abstractmethod
    def serialize(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[ToolDefinition]] = None
    ) -> str:
        """
        Serialize messages to a format suitable for the LLM provider.
        
        Args:
            messages: List of LangChain messages
            tools: Optional list of tool definitions
            
        Returns:
            Serialized string representation
        """
        ...


class StringSerializer(MessageSerializer):
    """Simple string concatenation serializer (current implementation)."""
    
    def serialize(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[ToolDefinition]] = None
    ) -> str:
        """Serialize messages to a single prompt string."""
        parts = []
        
        # Add tool definitions if available
        if tools:
            tool_descriptions = []
            for tool in tools:
                tool_desc = f"- {tool.name}: {tool.description}"
                if tool.parameters:
                    params = tool.parameters.get("properties", {})
                    if params:
                        param_str = ", ".join([
                            f"{k}: {v.get('type', 'string')}"
                            for k, v in params.items()
                        ])
                        tool_desc += f"\n  Parameters: {param_str}"
                tool_descriptions.append(tool_desc)
            
            if tool_descriptions:
                parts.append("Available tools:")
                parts.extend(tool_descriptions)
                parts.append("\nIMPORTANT: When you need to use a tool, you MUST call it.")
                parts.append("To call a tool, respond with a JSON object containing 'name' and 'arguments'.")
                parts.append("Example: {\"name\": \"rag_search\", \"arguments\": {\"query\": \"test\"}}")
                parts.append("If you need information, ALWAYS call the appropriate tool first before answering.")
                parts.append("")
        
        # Add messages
        for msg in messages:
            if isinstance(msg, SystemMessage):
                parts.append(f"System: {msg.content}")
            elif isinstance(msg, HumanMessage):
                parts.append(msg.content if isinstance(msg.content, str) else str(msg.content))
            elif isinstance(msg, AIMessage):
                # Handle AIMessage with tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    parts.append(f"Assistant (with tool calls): {msg.content}")
                else:
                    parts.append(f"Assistant: {msg.content}")
            elif hasattr(msg, "content"):
                parts.append(f"{type(msg).__name__}: {msg.content}")
            else:
                parts.append(str(msg))
        
        return "\n\n".join(parts)


class JSONSerializer(MessageSerializer):
    """JSON format serializer for providers that support structured messages."""
    
    def serialize(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[ToolDefinition]] = None
    ) -> str:
        """Serialize messages to JSON format."""
        message_list = []
        
        for msg in messages:
            msg_dict = {
                "role": self._get_role(msg),
                "content": self._get_content(msg)
            }
            
            # Add tool calls if present
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "name": tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", ""),
                        "arguments": tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                    }
                    for tc in msg.tool_calls
                ]
            
            message_list.append(msg_dict)
        
        result = {"messages": message_list}
        
        if tools:
            result["tools"] = [tool.to_dict() for tool in tools]
        
        return json.dumps(result, indent=2)
    
    def _get_role(self, msg: BaseMessage) -> str:
        """Get role from message type."""
        if isinstance(msg, SystemMessage):
            return "system"
        elif isinstance(msg, HumanMessage):
            return "user"
        elif isinstance(msg, AIMessage):
            return "assistant"
        else:
            return "user"
    
    def _get_content(self, msg: BaseMessage) -> str:
        """Extract content from message."""
        if hasattr(msg, "content"):
            content = msg.content
            return content if isinstance(content, str) else str(content)
        return str(msg)


class NativeSerializer(MessageSerializer):
    """Native format serializer (preserves LangChain message objects)."""
    
    def serialize(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[ToolDefinition]] = None
    ) -> str:
        """
        Serialize for native providers that accept LangChain messages directly.
        
        Note: This returns a placeholder - native serializers typically
        work with message objects directly, not strings.
        """
        # For native providers, messages are passed as-is
        # This is a placeholder for providers that need string representation
        return StringSerializer().serialize(messages, tools)


class MessageSerializerFactory:
    """Factory for creating message serializers."""
    
    @staticmethod
    def create_serializer(serializer_type: str = "string") -> MessageSerializer:
        """
        Create a message serializer based on type.
        
        Args:
            serializer_type: Type of serializer ("string", "json", "native")
            
        Returns:
            MessageSerializer instance
        """
        serializers = {
            "string": StringSerializer,
            "json": JSONSerializer,
            "native": NativeSerializer,
        }
        
        serializer_class = serializers.get(serializer_type.lower(), StringSerializer)
        return serializer_class()
