"""MCP (Model Context Protocol) client bridge: connect to MCP server, list and call tools."""

from typing import Any, Dict, List, Optional

import asyncio


class MCPClientBridge:
    """
    Bridge to an MCP server over stdio.
    Exposes list_tools and call_tool for use by LLM or other clients.
    """

    def __init__(
        self,
        command: str = "python",
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        """
        command: executable (e.g. "python").
        args: list of arguments (e.g. ["path/to/mcp_server.py"]).
        env: optional env vars for the subprocess.
        """
        self._command = command
        self._args = args or []
        self._env = env or {}

    def list_tools(self) -> List[Dict[str, Any]]:
        """List tools exposed by the MCP server."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            return []

        async def _run():
            params = StdioServerParameters(
                command=self._command,
                args=self._args,
                env=self._env,
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    return [
                        {"name": t.name, "description": getattr(t, "description", "") or ""}
                        for t in result.tools
                    ]

        try:
            return asyncio.run(_run())
        except Exception:
            return []

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call a tool by name with optional arguments."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            return {"error": "MCP SDK not installed"}

        async def _run():
            params = StdioServerParameters(
                command=self._command,
                args=self._args,
                env=self._env,
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments or {})
                    content = getattr(result, "content", []) or []
                    texts = [getattr(c, "text", str(c)) for c in content]
                    return {"result": "\n".join(texts) if texts else str(result)}

        try:
            return asyncio.run(_run())
        except Exception as e:
            return {"error": str(e)}
