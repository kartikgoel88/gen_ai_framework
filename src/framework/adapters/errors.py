"""Custom exceptions for LangChain adapter and agent operations."""


class ToolCallError(Exception):
    """Base exception for tool calling errors."""
    pass


class ToolCallParseError(ToolCallError):
    """Raised when tool call parsing fails."""
    
    def __init__(self, message: str, raw_response: str = ""):
        super().__init__(message)
        self.raw_response = raw_response


class ToolNotFoundError(ToolCallError):
    """Raised when a requested tool is not found."""
    
    def __init__(self, tool_name: str, available_tools: list[str] = None):
        message = f"Tool '{tool_name}' not found"
        if available_tools:
            message += f". Available tools: {', '.join(available_tools)}"
        super().__init__(message)
        self.tool_name = tool_name
        self.available_tools = available_tools or []


class InvalidToolArgumentsError(ToolCallError):
    """Raised when tool arguments are invalid."""
    
    def __init__(self, tool_name: str, arguments: dict, reason: str = ""):
        message = f"Invalid arguments for tool '{tool_name}': {arguments}"
        if reason:
            message += f". Reason: {reason}"
        super().__init__(message)
        self.tool_name = tool_name
        self.arguments = arguments
        self.reason = reason


class MessageConversionError(Exception):
    """Raised when message conversion fails."""
    
    def __init__(self, message: str, message_type: str = ""):
        super().__init__(message)
        self.message_type = message_type


class AdapterError(Exception):
    """Base exception for adapter errors."""
    pass


class StreamingNotSupportedError(AdapterError):
    """Raised when streaming is not supported."""
    pass
