"""Tool call interpretation layer - Agent responsibility.

This handles everything related to interpreting and executing tool calls:
- Parsing tool calls from LLM responses
- Validating tool calls
- Executing tools
- Error recovery
- Tool strategy selection

This is the "What do we do with that?" layer.
"""

from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

from ..llm.tool_calling import ToolCall
from ..adapters import (
    ToolCallParserFactory,
    ToolCallParseError,
    ToolNotFoundError,
    InvalidToolArgumentsError,
    ErrorRecoveryManager,
)
from .tool_registry import ToolRegistry


class ToolInterpreter:
    """
    Interprets tool calls from LLM responses and executes them.
    
    This is agent-layer responsibility - the adapter just returns what the LLM said.
    """
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        parser_type: str = "json",
        enable_recovery: bool = True
    ):
        """
        Initialize tool interpreter.
        
        Args:
            tool_registry: Registry of available tools
            parser_type: Type of parser to use ("json", "structured", "native")
            enable_recovery: Whether to enable error recovery
        """
        self._tool_registry = tool_registry
        self._parser = ToolCallParserFactory.create_parser(parser_type)
        self._recovery_manager = ErrorRecoveryManager() if enable_recovery else None
    
    def interpret_response(
        self,
        llm_message: AIMessage,
        available_tools: Optional[List[BaseTool]] = None
    ) -> Dict[str, Any]:
        """
        Interpret LLM response and extract tool calls.
        
        Args:
            llm_message: AIMessage from LLM
            available_tools: Optional list of tools (if None, uses registry)
            
        Returns:
            Dict with:
                - "has_tool_calls": bool
                - "tool_calls": List[ToolCall] if any
                - "content": str (remaining content after tool calls extracted)
        """
        # If LLM natively returned tool calls, use them directly
        if hasattr(llm_message, "tool_calls") and llm_message.tool_calls:
            return self._handle_native_tool_calls(llm_message)
        
        # Otherwise, parse tool calls from text response
        return self._parse_text_tool_calls(llm_message, available_tools)
    
    def _handle_native_tool_calls(self, llm_message: AIMessage) -> Dict[str, Any]:
        """Handle native tool calls from LLM message."""
        tool_calls = [
            self._extract_tool_call_from_dict(tc)
            for tc in llm_message.tool_calls
        ]
        
        return {
            "has_tool_calls": True,
            "tool_calls": tool_calls,
            "content": llm_message.content or ""
        }
    
    def _extract_tool_call_from_dict(self, tc: Any) -> ToolCall:
        """Extract ToolCall from dict or object."""
        if isinstance(tc, dict):
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {})
            tool_id = tc.get("id")
        else:
            tool_name = getattr(tc, "name", "")
            tool_args = getattr(tc, "args", {})
            tool_id = getattr(tc, "id", None)
        
        return ToolCall(
            name=tool_name,
            arguments=tool_args,
            call_id=tool_id
        )
    
    def _parse_text_tool_calls(
        self,
        llm_message: AIMessage,
        available_tools: Optional[List[BaseTool]]
    ) -> Dict[str, Any]:
        """Parse tool calls from text response."""
        content = llm_message.content or ""
        if not content:
            return {
                "has_tool_calls": False,
                "tool_calls": [],
                "content": ""
            }
        
        tool_names = self._get_tool_names(available_tools)
        
        # Parse tool calls from text
        try:
            parsed_tool_calls = self._parser.parse(content, tool_names)
            
            if parsed_tool_calls:
                remaining_content = self._remove_tool_call_json(content, parsed_tool_calls)
                return {
                    "has_tool_calls": True,
                    "tool_calls": parsed_tool_calls,
                    "content": remaining_content
                }
        except Exception as e:
            # Try error recovery if enabled
            recovery_result = self._try_recovery(e, content)
            if recovery_result:
                return recovery_result
        
        # No tool calls found
        return {
            "has_tool_calls": False,
            "tool_calls": [],
            "content": content
        }
    
    def _get_tool_names(self, available_tools: Optional[List[BaseTool]]) -> List[str]:
        """Get list of available tool names."""
        if available_tools:
            return [t.name for t in available_tools]
        return [t.name for t in self._tool_registry.get_all_tools()]
    
    def _remove_tool_call_json(self, content: str, tool_calls: List[ToolCall]) -> str:
        """Remove tool call JSON from content."""
        import json
        remaining_content = content
        for tc in tool_calls:
            try:
                json_str = json.dumps({"name": tc.name, "arguments": tc.arguments})
                remaining_content = remaining_content.replace(json_str, "").strip()
            except Exception:
                pass
        return remaining_content
    
    def _try_recovery(self, error: Exception, content: str) -> Optional[Dict[str, Any]]:
        """Try to recover from parsing error."""
        if not self._recovery_manager:
            return None
        
        try:
            recovery_result = self._recovery_manager.handle_error(
                ToolCallParseError(str(error), content),
                context={"retry_count": 0}
            )
            if recovery_result and isinstance(recovery_result, dict) and "name" in recovery_result:
                return {
                    "has_tool_calls": True,
                    "tool_calls": [ToolCall(
                        name=recovery_result.get("name"),
                        arguments=recovery_result.get("arguments", {}),
                        call_id=None
                    )],
                    "content": ""
                }
        except Exception:
            pass
        
        return None
    
    def validate_tool_call(self, tool_call: ToolCall) -> bool:
        """
        Validate a tool call.
        
        Args:
            tool_call: ToolCall to validate
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ToolNotFoundError if tool doesn't exist
            InvalidToolArgumentsError if arguments are invalid
        """
        # Check if tool exists
        self._check_tool_exists(tool_call.name)
        
        # Get tool for validation
        tool = self._tool_registry.get_tool(tool_call.name)
        if not tool:
            raise ToolNotFoundError(tool_call.name)
        
        # Validate arguments against tool schema
        self._validate_tool_arguments(tool, tool_call)
        
        return True
    
    def _check_tool_exists(self, tool_name: str) -> None:
        """Check if tool exists in registry."""
        if not self._tool_registry.has_tool(tool_name):
            raise ToolNotFoundError(
                tool_name,
                available_tools=[t.name for t in self._tool_registry.get_all_tools()]
            )
    
    def _validate_tool_arguments(self, tool: BaseTool, tool_call: ToolCall) -> None:
        """Validate tool call arguments against schema."""
        if not (hasattr(tool, "args_schema") and tool.args_schema):
            return
        
        try:
            schema = tool.args_schema.schema() if hasattr(tool.args_schema, "schema") else {}
            required = schema.get("required", [])
            properties = schema.get("properties", {})
            
            # Check required fields
            self._check_required_fields(tool_call, required)
            
            # Validate types
            self._validate_argument_types(tool_call, properties)
        except (InvalidToolArgumentsError, ToolNotFoundError):
            raise
        except Exception:
            # Schema validation failed, but don't block execution
            pass
    
    def _check_required_fields(self, tool_call: ToolCall, required: List[str]) -> None:
        """Check that all required fields are present."""
        for field in required:
            if field not in tool_call.arguments:
                raise InvalidToolArgumentsError(
                    tool_call.name,
                    tool_call.arguments,
                    reason=f"Missing required field: {field}"
                )
    
    def _validate_argument_types(self, tool_call: ToolCall, properties: dict) -> None:
        """Validate argument types against schema."""
        for field, value in tool_call.arguments.items():
            if field not in properties:
                continue
            
            expected_type = properties[field].get("type")
            if expected_type == "string" and not isinstance(value, str):
                raise InvalidToolArgumentsError(
                    tool_call.name,
                    tool_call.arguments,
                    reason=f"Field {field} should be string, got {type(value).__name__}"
                )
            elif expected_type == "integer" and not isinstance(value, int):
                raise InvalidToolArgumentsError(
                    tool_call.name,
                    tool_call.arguments,
                    reason=f"Field {field} should be integer, got {type(value).__name__}"
                )
    
    def execute_tool_call(
        self,
        tool_call: ToolCall,
        tool_call_id: Optional[str] = None
    ) -> ToolMessage:
        """
        Execute a tool call and return result.
        
        Args:
            tool_call: ToolCall to execute
            tool_call_id: Optional call ID for tracking
            
        Returns:
            ToolMessage with result
            
        Raises:
            ToolNotFoundError if tool doesn't exist
            InvalidToolArgumentsError if arguments are invalid
        """
        # Validate first
        self.validate_tool_call(tool_call)
        
        # Get tool
        tool = self._tool_registry.get_tool(tool_call.name)
        if not tool:
            raise ToolNotFoundError(tool_call.name)
        
        # Execute tool
        try:
            result = tool.invoke(tool_call.arguments)
            
            # Convert result to string if needed
            if not isinstance(result, str):
                import json
                try:
                    result = json.dumps(result)
                except:
                    result = str(result)
            
            return ToolMessage(
                content=result,
                name=tool_call.name,
                tool_call_id=tool_call_id or tool_call.call_id or f"call_{tool_call.name}"
            )
        except Exception as e:
            # Return error as tool message
            error_msg = f"Error executing {tool_call.name}: {str(e)}"
            return ToolMessage(
                content=error_msg,
                name=tool_call.name,
                tool_call_id=tool_call_id or tool_call.call_id or f"call_{tool_call.name}"
            )
    
    def process_llm_response(
        self,
        llm_message: AIMessage,
        available_tools: Optional[List[BaseTool]] = None
    ) -> List[BaseMessage]:
        """
        Process LLM response: interpret tool calls, execute them, return messages.
        
        This is the main entry point for the agent to process LLM responses.
        
        Args:
            llm_message: AIMessage from LLM
            available_tools: Optional list of tools
            
        Returns:
            List of messages: [AIMessage (possibly with tool_calls), ToolMessages...]
        """
        interpretation = self.interpret_response(llm_message, available_tools)
        
        if interpretation["has_tool_calls"]:
            return self._process_tool_calls(interpretation)
        else:
            return [AIMessage(content=interpretation["content"])]
    
    def _process_tool_calls(self, interpretation: Dict[str, Any]) -> List[BaseMessage]:
        """Process tool calls and return messages."""
        messages = []
        
        # Create AIMessage with tool calls
        tool_calls_langchain = self._convert_to_langchain_format(interpretation["tool_calls"])
        messages.append(AIMessage(
            content=interpretation["content"],
            tool_calls=tool_calls_langchain
        ))
        
        # Execute tool calls
        for tc in interpretation["tool_calls"]:
            tool_message = self._execute_with_error_handling(tc)
            messages.append(tool_message)
        
        return messages
    
    def _convert_to_langchain_format(self, tool_calls: List[ToolCall]) -> List[dict]:
        """Convert ToolCall objects to LangChain format."""
        return [
            {
                "name": tc.name,
                "args": tc.arguments,
                "id": tc.call_id or f"call_{tc.name}_{abs(hash(str(tc.arguments)))}"
            }
            for tc in tool_calls
        ]
    
    def _execute_with_error_handling(self, tool_call: ToolCall) -> ToolMessage:
        """Execute tool call with error handling."""
        try:
            return self.execute_tool_call(tool_call)
        except Exception as e:
            return ToolMessage(
                content=f"Error: {str(e)}",
                name=tool_call.name,
                tool_call_id=tool_call.call_id or f"call_{tool_call.name}"
            )
