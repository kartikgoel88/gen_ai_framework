"""Integration dependencies for FastAPI.

This module provides dependency injection functions for external integrations
like Confluence and MCP (Model Context Protocol).
"""

import json
from typing import Annotated

from fastapi import Depends

from ..confluence.client import ConfluenceClient
from ..mcp.client import MCPClientBridge
from ..config import get_settings_dep, FrameworkSettings


def get_confluence_client(
    settings: Annotated[FrameworkSettings, Depends(get_settings_dep)],
) -> ConfluenceClient | None:
    """Dependency that returns the Confluence client when configured.
    
    Returns None if CONFLUENCE_BASE_URL is not set, allowing optional
    Confluence integration.
    
    Supports both Cloud (email + API token) and Server (username + password)
    authentication methods.
    
    Args:
        settings: Framework settings (injected via FastAPI Depends)
        
    Returns:
        ConfluenceClient instance or None if not configured
        
    Example:
        ```python
        @app.post("/confluence/ingest")
        def ingest_confluence(
            space_key: str,
            confluence: ConfluenceClient | None = Depends(get_confluence_client)
        ):
            if not confluence:
                raise HTTPException(400, "Confluence not configured")
            return confluence.fetch_pages_for_ingest(space_key=space_key)
        ```
    """
    base_url = getattr(settings, "CONFLUENCE_BASE_URL", None) or ""
    if not base_url.strip():
        return None
    email = getattr(settings, "CONFLUENCE_EMAIL", None)
    api_token = getattr(settings, "CONFLUENCE_API_TOKEN", None)
    username = getattr(settings, "CONFLUENCE_USER", None)
    password = getattr(settings, "CONFLUENCE_PASSWORD", None)
    return ConfluenceClient(
        base_url=base_url,
        email=email,
        api_token=api_token,
        username=username,
        password=password,
    )


def get_mcp_client(
    settings: Annotated[FrameworkSettings, Depends(get_settings_dep)],
) -> MCPClientBridge:
    """Dependency that returns the MCP client bridge.
    
    MCP (Model Context Protocol) allows agents to use external tools
    via stdio-based servers.
    
    Args:
        settings: Framework settings (injected via FastAPI Depends)
        
    Returns:
        MCPClientBridge instance
        
    Example:
        ```python
        @app.get("/mcp/tools")
        def list_mcp_tools(mcp: MCPClientBridge = Depends(get_mcp_client)):
            return mcp.list_tools()
        ```
    """
    command = settings.MCP_COMMAND or "python"
    args = []
    if settings.MCP_ARGS:
        try:
            args = json.loads(settings.MCP_ARGS)
        except Exception:
            pass
    return MCPClientBridge(command=command, args=args, env={})
