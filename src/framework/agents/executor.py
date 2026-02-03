"""Agent executor abstraction layer."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage


class AgentExecutor(ABC):
    """Abstract interface for agent executors."""
    
    @abstractmethod
    def invoke(self, messages: List[BaseMessage], **kwargs: Any) -> Dict[str, Any]:
        """
        Execute agent with messages.
        
        Args:
            messages: List of LangChain messages
            **kwargs: Additional execution parameters
            
        Returns:
            Dict with "output" (str) and "messages" (list)
        """
        ...


class LangChainAgentExecutor(AgentExecutor):
    """Executor using LangChain's create_agent."""
    
    def __init__(self, graph: Any):
        """
        Initialize with a LangChain agent graph.
        
        Args:
            graph: Compiled LangChain agent graph
        """
        self._graph = graph
    
    def invoke(self, messages: List[BaseMessage], **kwargs: Any) -> Dict[str, Any]:
        """Execute using LangChain graph."""
        result = self._graph.invoke({"messages": messages}, **kwargs)
        out_messages = result.get("messages", [])
        
        # Extract output from last message
        output = ""
        if out_messages:
            last = out_messages[-1]
            if hasattr(last, "content") and last.content:
                output = last.content if isinstance(last.content, str) else str(last.content)
        
        return {"output": output, "messages": out_messages}
