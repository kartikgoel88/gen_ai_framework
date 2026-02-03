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
    
    Uses LangChain's create_agent which returns a LangGraph graph compatible
    with the framework's graph workflow.
    
    Args:
        llm: LangChain chat model (e.g. ChatOpenAI).
        rag: Optional RAGClient for RAG search tool.
        mcp_client: Optional MCPClientBridge for MCP tools.
        system_prompt: Optional system prompt for the agent.
        
    Returns:
        Compiled LangGraph graph that can be invoked with {"messages": [...]}
    """
    from langchain.agents import create_agent
    
    tools = build_framework_tools(
        rag_client=rag,
        mcp_client=mcp_client or _dummy_mcp(),
        enable_web_search=True
    )
    
    # Note: system_prompt should be included in messages when invoking the graph
    # Create agent graph using LangChain's create_agent
    # This returns a LangGraph graph that can be invoked with {"messages": [...]}
    graph = create_agent(
        model=llm,
        tools=tools,
    )
    
    return graph


def _dummy_mcp():
    """Return a no-op MCP client when none is provided."""
    class Dummy:
        def list_tools(self):
            return []
        def call_tool(self, name, arguments):
            return {"result": "MCP not configured"}
    return Dummy()
