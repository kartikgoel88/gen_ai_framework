"""LangGraph agent workflow: ReAct-style agent with tools (optional RAG + MCP)."""

from typing import Any, Optional

from ..rag.base import RAGClient
from ..agents.tools import build_framework_tools


def build_agent_graph(
    llm: Any,
    rag: Optional[RAGClient] = None,
    mcp_client: Any = None,
    system_prompt: Optional[str] = None,
):
    """
    Build a LangGraph ReAct agent with optional RAG and MCP tools.
    llm: LangChain chat model (e.g. ChatOpenAI).
    rag: Optional RAGClient for RAG search tool.
    mcp_client: Optional MCPClientBridge for MCP tools.
    Returns compiled graph.
    """
    try:
        from langgraph.prebuilt import create_react_agent
    except ImportError:
        try:
            from langgraph.prebuilt.chat_agent_executor import create_react_agent
        except ImportError:
            create_react_agent = None
    if create_react_agent is None:
        raise ImportError("LangGraph create_react_agent not found. Install langgraph.")
    tools = build_framework_tools(rag_client=rag, mcp_client=mcp_client or _dummy_mcp(), enable_web_search=True)
    system_prompt = system_prompt or (
        "You are a helpful AI assistant. Use the available tools when needed to answer questions. Be concise and accurate."
    )
    from langchain_core.messages import SystemMessage
    def state_modifier(state):
        return [SystemMessage(content=system_prompt)] + list(state.get("messages", []))
    graph = create_react_agent(llm, tools, state_modifier=state_modifier)
    return graph


def _dummy_mcp():
    """Return a no-op MCP client when none is provided."""
    class Dummy:
        def list_tools(self):
            return []
        def call_tool(self, name, arguments):
            return {"result": "MCP not configured"}
    return Dummy()
