"""Streaming agent with intermediate step visibility.

This module provides an agent implementation that streams intermediate
steps, tool calls, and reasoning in real-time.
"""

from typing import Any, Iterator, Optional
from enum import Enum
from dataclasses import dataclass

from langchain_core.tools import BaseTool

from .base import AgentBase
from .langchain_agent import LangChainReActAgent


class AgentEventType(Enum):
    """Types of events emitted during agent execution."""
    THINKING = "thinking"
    TOOL_SELECTION = "tool_selection"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    REASONING = "reasoning"
    RESPONSE_CHUNK = "response_chunk"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class AgentEvent:
    """Event emitted during agent execution."""
    type: AgentEventType
    content: str
    metadata: dict[str, Any] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata or {}
        }


class StreamingAgent(AgentBase):
    """Agent that streams intermediate steps and reasoning.
    
    This agent extends LangChainReActAgent to provide visibility into
    the agent's decision-making process by streaming events for:
    - Thinking/reasoning steps
    - Tool selection
    - Tool execution
    - Response generation
    
    Example:
        ```python
        agent = StreamingAgent(llm=llm, tools=tools)
        
        for event in agent.invoke_stream("What is RAG?"):
            print(f"[{event.type.value}] {event.content}")
        ```
    """
    
    def __init__(
        self,
        llm: Any,
        tools: list[BaseTool],
        system_prompt: Optional[str] = None,
        verbose: bool = False,
    ):
        """Initialize streaming agent.
        
        Args:
            llm: LangChain chat model
            tools: List of tools available to the agent
            system_prompt: Optional system prompt
            verbose: Enable verbose logging
        """
        self._base_agent = LangChainReActAgent(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            verbose=verbose
        )
        self._llm = llm
        self._tools = tools
    
    def invoke(self, message: str, **kwargs: Any) -> str:
        """Invoke agent and return final response (non-streaming).
        
        For streaming, use invoke_stream() instead.
        """
        return self._base_agent.invoke(message, **kwargs)
    
    def invoke_stream(
        self,
        message: str,
        *,
        system_prompt: Optional[str] = None,
        chat_history: Optional[list] = None,
        **kwargs: Any,
    ) -> Iterator[AgentEvent]:
        """Invoke agent and stream intermediate steps.
        
        Yields AgentEvent objects representing different stages of
        agent execution.
        
        Args:
            message: User message
            system_prompt: Optional system prompt override
            chat_history: Optional chat history
            **kwargs: Additional arguments
            
        Yields:
            AgentEvent objects for each step
        """
        try:
            # Emit thinking event
            yield AgentEvent(
                type=AgentEventType.THINKING,
                content="Analyzing the question and determining the best approach..."
            )
            
            # Build input text (same format as LangChainReActAgent.invoke)
            prompt_used = system_prompt or self._base_agent._system_prompt or ""
            if chat_history:
                history_str = self._base_agent._format_chat_history(chat_history)
                input_text = f"{prompt_used}\n\nChat history:\n{history_str}\n\nQuestion: {message}" if prompt_used else f"Chat history:\n{history_str}\n\nQuestion: {message}"
            else:
                input_text = f"{prompt_used}\n\nQuestion: {message}" if prompt_used else message

            # AgentExecutor doesn't stream; use invoke with return_intermediate_steps
            result = self._base_agent._executor.invoke(
                {"input": input_text},
                return_intermediate_steps=True,
                **kwargs,
            )
            intermediate_steps = result.get("intermediate_steps", [])
            for action, observation in intermediate_steps:
                tool_name = getattr(action, "tool", action) if hasattr(action, "tool") else str(action)
                tool_input = getattr(action, "tool_input", {}) if hasattr(action, "tool_input") else {}
                yield AgentEvent(
                    type=AgentEventType.TOOL_SELECTION,
                    content=f"Selected tool: {tool_name}",
                    metadata={"tool_name": tool_name},
                )
                yield AgentEvent(
                    type=AgentEventType.TOOL_CALL,
                    content=f"Calling {tool_name} with args: {tool_input}",
                    metadata={"tool_call": {"name": tool_name, "args": tool_input}},
                )
                obs_str = str(observation)[:500] + "..." if len(str(observation)) > 500 else str(observation)
                yield AgentEvent(
                    type=AgentEventType.TOOL_RESULT,
                    content=obs_str,
                    metadata={"full_content": str(observation)},
                )
            output = result.get("output", "")
            if output:
                chunk_size = 50
                for i in range(0, len(output), chunk_size):
                    yield AgentEvent(
                        type=AgentEventType.RESPONSE_CHUNK,
                        content=output[i : i + chunk_size],
                    )
            yield AgentEvent(
                type=AgentEventType.COMPLETE,
                content="Agent execution completed",
            )

        except Exception as e:
            yield AgentEvent(
                type=AgentEventType.ERROR,
                content=f"Error during agent execution: {str(e)}",
                metadata={"error_type": type(e).__name__}
            )
    
    def _process_chunk(self, chunk: dict[str, Any]) -> Iterator[AgentEvent]:
        """Process a chunk from LangGraph stream and emit events.
        
        Args:
            chunk: Chunk from graph stream
            
        Yields:
            AgentEvent objects
        """
        # Extract messages from chunk
        messages = chunk.get("messages", [])
        
        for msg in messages:
            # Check if it's a tool call
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    yield AgentEvent(
                        type=AgentEventType.TOOL_SELECTION,
                        content=f"Selected tool: {tool_call.get('name', 'unknown')}",
                        metadata={"tool_name": tool_call.get("name")}
                    )
                    yield AgentEvent(
                        type=AgentEventType.TOOL_CALL,
                        content=f"Calling {tool_call.get('name')} with args: {tool_call.get('args', {})}",
                        metadata={"tool_call": tool_call}
                    )
            
            # Check if it's a tool result
            if hasattr(msg, "content") and isinstance(msg.content, str):
                # Check if it looks like a tool result
                if "tool_result" in str(type(msg)).lower() or msg.content.startswith("Tool result:"):
                    yield AgentEvent(
                        type=AgentEventType.TOOL_RESULT,
                        content=msg.content[:200] + "..." if len(msg.content) > 200 else msg.content,
                        metadata={"full_content": msg.content}
                    )
                # Check if it's reasoning/thinking
                elif any(keyword in msg.content.lower() for keyword in ["think", "reason", "analyze", "consider"]):
                    yield AgentEvent(
                        type=AgentEventType.REASONING,
                        content=msg.content
                    )
                # Otherwise it's a response chunk
                else:
                    yield AgentEvent(
                        type=AgentEventType.RESPONSE_CHUNK,
                        content=msg.content
                    )
    
    def get_tools_description(self) -> list[dict[str, Any]]:
        """Return list of available tools."""
        return self._base_agent.get_tools_description()
