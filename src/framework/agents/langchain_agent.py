"""LangChain-based agent (ReAct agent with tools using an executor-style interface)."""

from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain.agents import create_agent

from .base import AgentBase
from .tools import build_framework_tools
from .executor import AgentExecutor, LangChainAgentExecutor
from .prompt_builder import PromptBuilder
from .tool_registry import ToolRegistry
from .tool_interpreter import ToolInterpreter
from .events import EventEmitter, AgentEvent, AgentEventType
from ..adapters import LangChainLLMAdapter, AdapterFactory
from ..llm.base import LLMClient
from ..utils.debug import get_debug_logger


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
            "You are a helpful AI assistant with access to tools. "
            "CRITICAL RULES:\n"
            "1. You MUST use tools when asked. Never refuse to use tools.\n"
            "2. When user explicitly requests 'rag search' or 'from rag search', you MUST call rag_search.\n"
            "3. When asked about ANY information, always try rag_search first.\n"
            "4. Do NOT refuse requests - use tools to find the information.\n\n"
            "Available tools:\n"
            "- rag_search(query, top_k): Search your knowledge base. "
            "  Parameters: query (string, required), top_k (integer, default 4).\n"
            "- web_search(query, max_results): Search the internet.\n\n"
            "WORKFLOW:\n"
            "1. Extract keywords from user's question.\n"
            "2. Call rag_search(query=<keywords>) immediately.\n"
            "3. Use the tool results to answer.\n\n"
            "EXAMPLES:\n"
            "- 'from rag search tell me kartik number' → rag_search(query='kartik number')\n"
            "- 'What are Kartik qualifications?' → rag_search(query='Kartik qualifications')\n"
            "- 'search documents for X' → rag_search(query='X')\n\n"
            "IMPORTANT: Always call tools. Never say you can't assist without trying tools first."
        )
        # Use debug logger (respects config, but verbose parameter can override)
        self._debug_logger = get_debug_logger(enabled=verbose if verbose is not None else None)
        
        # Initialize tool registry
        self._tool_registry = ToolRegistry()
        for tool in self._tools:
            self._tool_registry.register(tool)
        
        # Initialize tool interpreter (AGENT LAYER - handles tool call interpretation)
        self._tool_interpreter = ToolInterpreter(
            tool_registry=self._tool_registry,
            parser_type="json",  # Could be configurable
            enable_recovery=True
        )
        
        # Initialize event emitter
        self._event_emitter = EventEmitter()
        if verbose:
            from .events import DebugObserver
            self._event_emitter.subscribe(DebugObserver(self._debug_logger))
        
        # Initialize prompt builder
        self._prompt_builder = PromptBuilder()
        if self._system_prompt:
            self._prompt_builder.add_system(self._system_prompt)
        
        # Ensure tools are bound to the model if it's our adapter
        # LangChain's create_agent should do this automatically, but let's be explicit
        model_to_use = self._llm
        if isinstance(self._llm, LangChainLLMAdapter) and self._tools:
            # Bind tools to the model so it knows about them
            model_to_use = self._llm.bind_tools(self._tools)
            self._debug_logger.log(f"Bound {len(self._tools)} tools to LLM adapter", category="agent")
            self._event_emitter.emit(AgentEvent(
                event_type=AgentEventType.TOOL_SELECTION,
                content=f"Bound {len(self._tools)} tools",
                metadata={"tool_count": len(self._tools)}
            ))
        
        # Note: system_prompt is handled in invoke() method, not passed to create_agent
        self._graph = create_agent(
            model=model_to_use,
            tools=self._tools,
        )
        
        # Use executor abstraction
        self._executor_impl = LangChainAgentExecutor(self._graph)
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
        messages = self._build_messages(message, system_prompt, chat_history)
        
        self._emit_message_received(message, chat_history)
        
        result = self._invoke_messages(messages, **kwargs)
        output = result.get("output", "")
        
        self._emit_message_sent(output)
        
        return output if isinstance(output, str) else str(output)
    
    def _build_messages(
        self,
        message: str,
        system_prompt: Optional[str],
        chat_history: Optional[list]
    ) -> list[BaseMessage]:
        """Build messages from user input, system prompt, and chat history."""
        prompt_used = system_prompt if system_prompt is not None else self._system_prompt
        
        builder = PromptBuilder()
        if prompt_used:
            builder.add_system(prompt_used)
        if chat_history:
            builder.add_chat_history(chat_history)
        
        # Get tool definitions from registry
        tool_definitions = self._tool_registry.get_tool_definitions()
        if tool_definitions:
            builder.add_tools(tool_definitions)
        
        return builder.build_messages(message)
    
    def _emit_message_received(self, message: str, chat_history: Optional[list]) -> None:
        """Emit message received event."""
        self._event_emitter.emit(AgentEvent(
            event_type=AgentEventType.MESSAGE_RECEIVED,
            content=message,
            metadata={"has_history": bool(chat_history)}
        ))
    
    def _emit_message_sent(self, output: str) -> None:
        """Emit message sent event."""
        self._event_emitter.emit(AgentEvent(
            event_type=AgentEventType.MESSAGE_SENT,
            content=output[:100] if output else "",
            metadata={"output_length": len(output)}
        ))

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
        self._log_invocation_start(messages)
        
        result = self._executor_impl.invoke(messages, **kwargs)
        out_messages = result.get("messages", [])
        
        self._log_invocation_result(out_messages)
        
        output = self._extract_output(out_messages)
        return {"output": output, "messages": out_messages}
    
    def _log_invocation_start(self, messages: list[BaseMessage]) -> None:
        """Log the start of agent invocation."""
        self._debug_logger.log(f"Invoking agent with {len(messages)} messages", category="agent")
        self._debug_logger.log(f"Tools available: {[t.name for t in self._tools]}", category="agent")
        
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            content_preview = str(msg.content)[:150] if hasattr(msg, "content") and msg.content else "None"
            self._debug_logger.log_message(msg_type, content_preview, index=i)
    
    def _log_invocation_result(self, out_messages: list) -> None:
        """Log the result of agent invocation."""
        self._debug_logger.log(f"Agent returned {len(out_messages)} messages", category="agent")
        tool_calls_found = False
        
        for i, msg in enumerate(out_messages):
            if self._has_tool_calls(msg):
                tool_calls_found = True
                self._log_tool_calls(msg, i)
            elif type(msg).__name__ == "ToolMessage":
                tool_calls_found = True
                self._log_tool_message(msg, i)
            else:
                self._log_regular_message(msg, i)
        
        if not tool_calls_found:
            self._debug_logger.log_warning(
                "No tool calls detected in agent response! "
                "The agent may not be using tools. Check system prompt and tool bindings."
            )
    
    def _has_tool_calls(self, msg: Any) -> bool:
        """Check if message has tool calls."""
        return hasattr(msg, "tool_calls") and msg.tool_calls
    
    def _log_tool_calls(self, msg: Any, index: int) -> None:
        """Log tool calls in message."""
        msg_type = type(msg).__name__
        self._debug_logger.log(
            f"Message {index}: {msg_type} with {len(msg.tool_calls)} tool calls",
            category="agent"
        )
        
        for tc in msg.tool_calls:
            tool_name = tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown')
            tool_args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
            self._debug_logger.log_tool_call(tool_name, tool_args)
            
            self._event_emitter.emit(AgentEvent(
                event_type=AgentEventType.TOOL_CALL,
                content=f"Calling {tool_name}",
                metadata={"tool_name": tool_name, "arguments": tool_args}
            ))
    
    def _log_tool_message(self, msg: Any, index: int) -> None:
        """Log tool message."""
        tool_name = getattr(msg, "name", "unknown")
        tool_content = str(getattr(msg, "content", ""))[:200] if hasattr(msg, "content") else "None"
        self._debug_logger.log(
            f"Message {index}: ToolMessage from {tool_name} - {tool_content}...",
            category="tool"
        )
    
    def _log_regular_message(self, msg: Any, index: int) -> None:
        """Log regular message."""
        msg_type = type(msg).__name__
        if hasattr(msg, "content"):
            content_preview = str(msg.content)[:150] if msg.content else "None"
            self._debug_logger.log_message(msg_type, content_preview, index=index)
    
    def _extract_output(self, out_messages: list) -> str:
        """Extract output string from messages."""
        if not out_messages:
            return ""
        
        last = out_messages[-1]
        if hasattr(last, "content") and last.content:
            content = last.content
            return content if isinstance(content, str) else str(content)
        
        return ""

    def get_tools_description(self) -> list[dict[str, Any]]:
        """Return list of {name, description} for tools available to this agent."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
            }
            for tool in self._tools
        ]


class _ExecutorWrapper:
    """Thin wrapper that exposes an AgentExecutor-like API: invoke({"input": str}) -> {"output": str}."""

    def __init__(self, agent: LangChainReActAgent):
        self._agent = agent

    def invoke(self, inputs: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        # Support both old format (input as string) and new format (messages list)
        if "messages" in inputs:
            messages = inputs["messages"]
        else:
            input_text = inputs.get("input", "")
            # Fallback: if input contains system prompt, try to parse it
            messages: list[BaseMessage] = [HumanMessage(content=input_text)]
        return self._agent._invoke_messages(messages, **kwargs)


def create_tool_agent(
    llm: LLMClient,
    *,
    rag_client: Optional[Any] = None,
    mcp_client: Optional[Any] = None,
    enable_web_search: bool = True,
    system_prompt: Optional[str] = None,
    verbose: bool = False,
) -> LangChainReActAgent:
    """Create a tool-based agent with automatic tool setup.
    
    This factory function simplifies creating a ReAct agent with tools by:
    - Wrapping the framework LLMClient as a LangChain model
    - Building framework tools (RAG, MCP, web search) automatically
    - Configuring the agent with the provided options
    
    Args:
        llm: Framework LLMClient (from get_llm or LLMProviderRegistry.create)
        rag_client: Optional RAG client for document search tool
        mcp_client: Optional MCP client for additional tools
        enable_web_search: Whether to enable web search tool (default: True)
        system_prompt: Optional custom system prompt (default: helpful assistant)
        verbose: Whether to enable verbose logging (default: False)
        
    Returns:
        LangChainReActAgent instance ready to use
        
    Example:
        ```python
        from src.framework.api.deps import get_llm, get_rag
        from src.framework.agents import create_tool_agent
        
        settings = get_settings()
        llm = get_llm(settings)
        rag = get_rag(settings)
        
        agent = create_tool_agent(
            llm=llm,
            rag_client=rag,
            enable_web_search=True,
            system_prompt="You are a helpful research assistant.",
        )
        
        response = agent.invoke("What documents have you ingested?")
        ```
    """
    tools = build_framework_tools(
        rag_client=rag_client,
        mcp_client=mcp_client,
        enable_web_search=enable_web_search,
    )
    
    # Use adapter factory to get provider-specific adapter
    # Try to detect provider from LLM client type
    provider = "openai"  # Default
    if hasattr(llm, "__class__"):
        class_name = llm.__class__.__name__.lower()
        if "anthropic" in class_name or "claude" in class_name:
            provider = "anthropic"
        elif "gemini" in class_name or "google" in class_name:
            provider = "gemini"
        elif "openai" in class_name:
            provider = "openai"
    
    lc_llm = AdapterFactory.create(provider, llm, bound_tools=None)
    
    return LangChainReActAgent(
        llm=lc_llm,
        tools=tools,
        system_prompt=system_prompt,
        verbose=verbose,
    )
