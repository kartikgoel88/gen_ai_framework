"""LangChain-based agent (ReAct agent with tools using an executor-style interface)."""

from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain.agents import create_agent

from .base import AgentBase


class LangChainReActAgent(AgentBase):
    """Agent using LangChain create_agent with an executor-style invoke (input -> output)."""

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
        self._graph = create_agent(
            model=self._llm,
            tools=self._tools,
            prompt=self._system_prompt,
        )
        # Expose executor-style interface: .invoke({"input": ...}) -> {"output": ...}
        self._executor = _ExecutorWrapper(self)

    def invoke(
        self,
        message: str,
        *,
        system_prompt: Optional[str] = None,
        chat_history: Optional[list] = None,
        **kwargs: Any,
    ) -> str:
        """Run the agent with a user message and return the final reply.

        Uses the executor-style API: builds a single input (system prompt + optional
        chat history + question), calls the internal executor, and returns the
        assistant's output string.

        Args:
            message: The user's question or instruction.
            system_prompt: Optional override for the default system prompt.
            chat_history: Optional list of prior turns (dicts with "role" and "content",
                or objects with .type and .content). Prepended to the prompt when given.
            **kwargs: Passed through to the underlying graph invoke (e.g. config).

        Returns:
            The assistant's reply as a string. Empty string if no content.
        """
        prompt_used = system_prompt if system_prompt is not None else self._system_prompt
        if chat_history:
            history_str = self._format_chat_history(chat_history)
            input_text = f"{prompt_used}\n\nChat history:\n{history_str}\n\nQuestion: {message}" if prompt_used else f"Chat history:\n{history_str}\n\nQuestion: {message}"
        else:
            input_text = f"{prompt_used}\n\nQuestion: {message}" if prompt_used else message
        result = self._executor.invoke({"input": input_text}, **kwargs)
        output = result.get("output", "")
        return output if isinstance(output, str) else str(output)

    def _format_chat_history(self, chat_history: list) -> str:
        lines = []
        for msg in chat_history:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                lines.append(f"{role.capitalize()}: {content}")
            elif hasattr(msg, "type") and hasattr(msg, "content"):
                role = "user" if getattr(msg.type, "lower", lambda: "")() == "human" else "assistant"
                lines.append(f"{role.capitalize()}: {msg.content}")
        return "\n".join(lines)

    def _invoke_messages(self, messages: list[BaseMessage], **kwargs: Any) -> dict[str, Any]:
        """Run the agent graph with a list of messages and return output + messages.

        Called by _ExecutorWrapper when using the executor-style invoke({"input": ...}).
        The graph expects a "messages" key; this method forwards that and extracts
        the final assistant content as "output".

        Args:
            messages: List of LangChain message objects (e.g. HumanMessage, AIMessage).
            **kwargs: Passed through to the graph invoke (e.g. config).

        Returns:
            Dict with "output" (str, final assistant reply) and "messages" (list of
            messages from the graph, including the new assistant message).
        """
        result = self._graph.invoke({"messages": messages}, **kwargs)
        out_messages = result.get("messages", [])
        output = ""
        if out_messages:
            last = out_messages[-1]
            if hasattr(last, "content") and last.content:
                output = last.content if isinstance(last.content, str) else str(last.content)
        return {"output": output, "messages": out_messages}


class _ExecutorWrapper:
    """Thin wrapper that exposes an AgentExecutor-like API: invoke({"input": str}) -> {"output": str}."""

    def __init__(self, agent: LangChainReActAgent):
        self._agent = agent

    def invoke(self, inputs: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        input_text = inputs.get("input", "")
        # input_text is already full prompt (system + chat history + question) from public invoke()
        messages: list[BaseMessage] = [HumanMessage(content=input_text)]
        return self._agent._invoke_messages(messages, **kwargs)
