"""LangChain tools built from framework components (RAG, MCP, Web Search)."""

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
        tool_name = info.get("name", "unknown")
        tool_description = info.get("description", f"MCP tool: {tool_name}")
        mcp = mcp_client

        class MCPToolInput(BaseModel):
            arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments as JSON object")

        class MCPToolBound(BaseTool):
            name: str = tool_name
            description: str = tool_description
            args_schema: type[BaseModel] = MCPToolInput

            def _run(self, arguments: Optional[dict] = None) -> str:
                out = mcp.call_tool(tool_name, arguments or {})
                if "error" in out:
                    return f"Error: {out['error']}"
                return out.get("result", str(out))

        result.append(MCPToolBound())
    return result


def build_web_search_tool() -> BaseTool:
    """Build a LangChain tool for web search (supports LinkedIn profile search)."""
    
    class WebSearchInput(BaseModel):
        query: str = Field(description="Search query. For LinkedIn profiles, use format: 'linkedin profile [name]' or 'site:linkedin.com [name]'")
        max_results: int = Field(default=5, description="Maximum number of search results to return")
    
    class WebSearchTool(BaseTool):
        name: str = "web_search"
        description: str = (
            "Search the web for information. Use this to find LinkedIn profiles, current information, "
            "or any web content. For LinkedIn profiles, use queries like 'linkedin profile John Doe' "
            "or 'site:linkedin.com/in/ John Doe'. Returns search results with titles, URLs, and snippets."
        )
        args_schema: type[BaseModel] = WebSearchInput
        
        def _run(self, query: str, max_results: int = 5) -> str:
            try:
                from duckduckgo_search import DDGS
                
                # For LinkedIn profiles, optimize the query
                if "linkedin" in query.lower() or "linkedin profile" in query.lower():
                    # Ensure we're searching LinkedIn specifically
                    if "site:linkedin.com" not in query.lower():
                        query = f"site:linkedin.com/in/ {query.replace('linkedin profile', '').replace('linkedin', '').strip()}"
                
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=max_results))
                
                if not results:
                    return f"No search results found for: {query}"
                
                formatted_results = []
                for i, result in enumerate(results, 1):
                    title = result.get("title", "No title")
                    url = result.get("href", "No URL")
                    body = result.get("body", "No description")
                    formatted_results.append(f"{i}. **{title}**\n   URL: {url}\n   {body}\n")
                
                return "\n".join(formatted_results)
            except ImportError:
                return (
                    "Web search is not available. Install duckduckgo-search: "
                    "pip install duckduckgo-search"
                )
            except Exception as e:
                return f"Error during web search: {str(e)}"
    
    return WebSearchTool()


def build_framework_tools(
    rag_client: Optional[Any] = None,
    mcp_client: Optional[Any] = None,
    enable_web_search: bool = True,
) -> list[BaseTool]:
    """Build all framework tools for the agent (RAG + MCP + Web Search)."""
    tools: list[BaseTool] = []
    if rag_client is not None:
        tools.append(build_rag_tool(rag_client))
    if enable_web_search:
        tools.append(build_web_search_tool())
    if mcp_client is not None:
        tools.extend(build_mcp_tools(mcp_client))
    return tools
