"""LangChain-based agent (LangGraph ReAct agent with tools)."""

from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from .base import AgentBase


class LangChainReActAgent(AgentBase):
    """Agent using LangGraph create_react_agent (ReAct-style tool-calling loop)."""

    def __init__(
        self,
        llm: Any,
        tools: list[BaseTool],
        system_prompt: Optional[str] = None,
        verbose: bool = False,
    ):
        self._llm = llm
        self._tools = tools
        self._system_prompt = system_prompt or (
            "You are a helpful AI assistant. Use the available tools when needed to answer questions. "
            "Be concise and accurate."
        )
        self._verbose = verbose
        self._graph = create_react_agent(
            model=self._llm,
            tools=self._tools,
            prompt=self._system_prompt,
        )

    def invoke(
        self,
        message: str,
        *,
        system_prompt: Optional[str] = None,
        chat_history: Optional[list] = None,
        **kwargs: Any,
    ) -> str:
        messages: list[BaseMessage] = []
        if system_prompt is not None:
            messages.append(SystemMessage(content=system_prompt))
        elif self._system_prompt:
            messages.append(SystemMessage(content=self._system_prompt))
        if chat_history:
            for msg in chat_history:
                if isinstance(msg, BaseMessage):
                    messages.append(msg)
                elif isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=message))
        inputs = {"messages": messages}
        result = self._graph.invoke(inputs, **kwargs)
        out_messages = result.get("messages", [])
        if not out_messages:
            return ""
        last = out_messages[-1]
        if hasattr(last, "content") and last.content:
            return last.content if isinstance(last.content, str) else str(last.content)
        return ""

    def get_tools_description(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools
        ]
