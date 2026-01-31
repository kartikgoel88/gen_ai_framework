"""LangChain tools built from framework components (RAG, MCP)."""

from typing import Any, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


def build_rag_tool(rag_client: Any) -> BaseTool:
    """Build a LangChain tool that queries the RAG store."""
    rag = rag_client  # closure for _run

    class RAGQueryInput(BaseModel):
        query: str = Field(description="Search query to find relevant documents")
        top_k: int = Field(default=4, description="Number of chunks to retrieve")

    class RAGSearchTool(BaseTool):
        name: str = "rag_search"
        description: str = "Search the knowledge base for relevant documents. Use when you need to answer questions from ingested documents."
        args_schema: type[BaseModel] = RAGQueryInput

        def _run(self, query: str, top_k: int = 4) -> str:
            chunks = rag.retrieve(query, top_k=top_k)
            if not chunks:
                return "No relevant documents found."
            return "\n\n---\n\n".join(c["content"] for c in chunks)

    return RAGSearchTool()


def build_mcp_tools(mcp_client: Any) -> list[BaseTool]:
    """Build LangChain tools from MCP server tools."""

    tools_list = mcp_client.list_tools()
    if not tools_list:
        return []

    result = []
    for info in tools_list:
        name = info.get("name", "unknown")
        description = info.get("description", f"MCP tool: {name}")

        class MCPToolInput(BaseModel):
            arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments as JSON object")

        class MCPTool(BaseTool):
            name: str = name
            description: str = description
            args_schema: type[BaseModel] = MCPToolInput
            _mcp_client: Any = None
            _tool_name: str = name

            def _run(self, arguments: Optional[dict] = None) -> str:
                out = self._mcp_client.call_tool(self._tool_name, arguments or {})
                if "error" in out:
                    return f"Error: {out['error']}"
                return out.get("result", str(out))

        # Create a closure per tool so _tool_name is bound
        tool_name = name
        mcp = mcp_client

        class MCPToolBound(BaseTool):
            name: str = tool_name
            description: str = description
            args_schema: type[BaseModel] = MCPToolInput

            def _run(self, arguments: Optional[dict] = None) -> str:
                out = mcp.call_tool(tool_name, arguments or {})
                if "error" in out:
                    return f"Error: {out['error']}"
                return out.get("result", str(out))

        result.append(MCPToolBound())
    return result


def build_framework_tools(
    rag_client: Optional[Any] = None,
    mcp_client: Optional[Any] = None,
) -> list[BaseTool]:
    """Build all framework tools for the agent (RAG + MCP)."""
    tools: list[BaseTool] = []
    if rag_client is not None:
        tools.append(build_rag_tool(rag_client))
    if mcp_client is not None:
        tools.extend(build_mcp_tools(mcp_client))
    return tools
