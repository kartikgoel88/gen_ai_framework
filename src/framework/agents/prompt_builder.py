"""Prompt building system for agents."""

from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ..llm.tool_calling import ToolDefinition


class PromptBuilder:
    """Builder for constructing agent prompts."""
    
    def __init__(self):
        self._system_parts: List[str] = []
        self._tool_definitions: List[ToolDefinition] = []
        self._examples: List[str] = []
        self._chat_history: List[Dict[str, str]] = []
    
    def add_system(self, content: str) -> "PromptBuilder":
        """Add system prompt content."""
        self._system_parts.append(content)
        return self
    
    def add_tools(self, tools: List[ToolDefinition]) -> "PromptBuilder":
        """Add tool definitions."""
        self._tool_definitions.extend(tools)
        return self
    
    def add_examples(self, examples: List[str]) -> "PromptBuilder":
        """Add example interactions."""
        self._examples.extend(examples)
        return self
    
    def add_chat_history(self, history: List[Dict[str, str]]) -> "PromptBuilder":
        """Add chat history."""
        self._chat_history.extend(history)
        return self
    
    def build_messages(self, user_message: str) -> List[BaseMessage]:
        """
        Build LangChain messages from builder state.
        
        Args:
            user_message: The current user message
            
        Returns:
            List of BaseMessage objects
        """
        messages: List[BaseMessage] = []
        
        # Build system message
        system_msg = self._build_system_message()
        if system_msg:
            messages.append(system_msg)
        
        # Add user message with optional chat history
        user_msg = self._build_user_message(user_message)
        messages.append(user_msg)
        
        return messages
    
    def _build_system_message(self) -> Optional[SystemMessage]:
        """Build system message from builder state."""
        if not self._system_parts:
            return None
        
        system_content = "\n\n".join(self._system_parts)
        system_content = self._add_tools_to_system(system_content)
        system_content = self._add_examples_to_system(system_content)
        
        return SystemMessage(content=system_content)
    
    def _add_tools_to_system(self, content: str) -> str:
        """Add tool descriptions to system content."""
        if not self._tool_definitions:
            return content
        
        content += "\n\nAvailable tools:\n"
        for tool in self._tool_definitions:
            content += f"- {tool.name}: {tool.description}\n"
        
        return content
    
    def _add_examples_to_system(self, content: str) -> str:
        """Add examples to system content."""
        if not self._examples:
            return content
        
        content += "\n\nExamples:\n"
        for example in self._examples:
            content += f"{example}\n"
        
        return content
    
    def _build_user_message(self, user_message: str) -> HumanMessage:
        """Build user message with optional chat history."""
        if not self._chat_history:
            return HumanMessage(content=user_message)
        
        history_str = self._format_chat_history()
        return HumanMessage(content=f"Chat history:\n{history_str}\n\nQuestion: {user_message}")
    
    def _format_chat_history(self) -> str:
        """Format chat history as string."""
        return "\n".join([
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
            for msg in self._chat_history
        ])
    
    def build_string(self, user_message: str) -> str:
        """
        Build a single prompt string (for non-message-based LLMs).
        
        Args:
            user_message: The current user message
            
        Returns:
            Combined prompt string
        """
        parts = []
        
        # System content
        if self._system_parts:
            parts.append("System:")
            parts.extend(self._system_parts)
        
        # Tool definitions
        if self._tool_definitions:
            parts.append("\nAvailable tools:")
            for tool in self._tool_definitions:
                parts.append(f"- {tool.name}: {tool.description}")
        
        # Examples
        if self._examples:
            parts.append("\nExamples:")
            parts.extend(self._examples)
        
        # Chat history
        if self._chat_history:
            parts.append("\nChat history:")
            for msg in self._chat_history:
                parts.append(f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}")
        
        # User message
        parts.append(f"\nUser: {user_message}")
        
        return "\n".join(parts)
    
    def clear(self) -> "PromptBuilder":
        """Clear all builder state."""
        self._system_parts.clear()
        self._tool_definitions.clear()
        self._examples.clear()
        self._chat_history.clear()
        return self
