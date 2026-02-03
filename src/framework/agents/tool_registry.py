"""Dynamic tool registry for agents."""

from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool

from ..llm.tool_calling import ToolDefinition


class ToolRegistry:
    """Registry for managing available tools dynamically."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._tool_definitions: Dict[str, ToolDefinition] = {}
    
    def register(self, tool: BaseTool) -> None:
        """
        Register a tool.
        
        Args:
            tool: LangChain BaseTool instance
        """
        self._tools[tool.name] = tool
        
        # Create tool definition
        parameters = {}
        if hasattr(tool, "args_schema") and tool.args_schema:
            try:
                schema = tool.args_schema.schema() if hasattr(tool.args_schema, "schema") else {}
                parameters = schema
            except:
                pass
        
        self._tool_definitions[tool.name] = ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters=parameters
        )
    
    def unregister(self, tool_name: str) -> None:
        """
        Unregister a tool.
        
        Args:
            tool_name: Name of tool to remove
        """
        self._tools.pop(tool_name, None)
        self._tool_definitions.pop(tool_name, None)
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(tool_name)
    
    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def get_tool_definitions(self) -> List[ToolDefinition]:
        """Get all tool definitions."""
        return list(self._tool_definitions.values())
    
    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._tools
    
    def clear(self) -> None:
        """Clear all tools."""
        self._tools.clear()
        self._tool_definitions.clear()


class ToolSelector:
    """Strategy for selecting which tools to use."""
    
    def select(self, registry: ToolRegistry, context: Optional[Dict[str, Any]] = None) -> List[BaseTool]:
        """
        Select tools from registry.
        
        Args:
            registry: Tool registry
            context: Optional context for selection
            
        Returns:
            List of selected tools
        """
        raise NotImplementedError


class AllToolsSelector(ToolSelector):
    """Select all available tools."""
    
    def select(self, registry: ToolRegistry, context: Optional[Dict[str, Any]] = None) -> List[BaseTool]:
        """Return all tools."""
        return registry.get_all_tools()


class ConditionalToolSelector(ToolSelector):
    """Select tools based on conditions."""
    
    def __init__(self, condition: callable):
        """
        Initialize with a condition function.
        
        Args:
            condition: Function(tool_name, tool) -> bool
        """
        self.condition = condition
    
    def select(self, registry: ToolRegistry, context: Optional[Dict[str, Any]] = None) -> List[BaseTool]:
        """Select tools matching condition."""
        return [
            tool for name, tool in registry._tools.items()
            if self.condition(name, tool)
        ]


class NamedToolSelector(ToolSelector):
    """Select specific tools by name."""
    
    def __init__(self, tool_names: List[str]):
        """
        Initialize with list of tool names.
        
        Args:
            tool_names: List of tool names to select
        """
        self.tool_names = tool_names
    
    def select(self, registry: ToolRegistry, context: Optional[Dict[str, Any]] = None) -> List[BaseTool]:
        """Select tools by name."""
        return [
            registry.get_tool(name)
            for name in self.tool_names
            if registry.has_tool(name)
        ]
