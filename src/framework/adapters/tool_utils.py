"""Shared utilities for tool conversion and handling."""

from typing import List, Optional

from langchain_core.tools import BaseTool

from ..llm.tool_calling import ToolDefinition


def convert_tools_to_definitions(tools: Optional[List[BaseTool]]) -> List[ToolDefinition]:
    """
    Convert LangChain BaseTool objects to ToolDefinition objects.
    
    Args:
        tools: Optional list of LangChain BaseTool instances
        
    Returns:
        List of ToolDefinition objects
    """
    if not tools:
        return []
    
    tool_definitions = []
    for tool in tools:
        if hasattr(tool, "name") and hasattr(tool, "description"):
            # Extract parameter schema
            parameters = {}
            if hasattr(tool, "args_schema") and tool.args_schema:
                try:
                    schema = tool.args_schema.schema() if hasattr(tool.args_schema, "schema") else {}
                    parameters = schema
                except Exception:
                    pass
            
            tool_definitions.append(ToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=parameters
            ))
    
    return tool_definitions


def convert_tool_call_to_langchain_format(tool_call, tool_name: str = None, tool_args: dict = None, call_id: str = None) -> dict:
    """
    Convert a ToolCall object or dict to LangChain tool call format.
    
    Args:
        tool_call: ToolCall object or dict with tool call data
        tool_name: Optional tool name (if not in tool_call)
        tool_args: Optional tool arguments (if not in tool_call)
        call_id: Optional call ID (if not in tool_call)
        
    Returns:
        Dict in LangChain format: {"name": str, "args": dict, "id": str}
    """
    if isinstance(tool_call, dict):
        name = tool_call.get("name") or tool_name
        args = tool_call.get("args") or tool_call.get("arguments") or tool_args or {}
        call_id = tool_call.get("id") or call_id
    else:
        # Assume ToolCall object
        name = getattr(tool_call, "name", None) or tool_name
        args = getattr(tool_call, "arguments", None) or tool_args or {}
        call_id = getattr(tool_call, "call_id", None) or call_id
    
    if not name:
        raise ValueError("Tool name is required")
    
    # Generate call_id if not provided
    if not call_id:
        call_id = f"call_{name}_{abs(hash(str(args)))}"
    
    return {
        "name": name,
        "args": args,
        "id": call_id
    }
