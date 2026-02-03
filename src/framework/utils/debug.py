"""Debug logging utility for the framework."""

import sys
from typing import Optional

from ..config import get_settings


class DebugLogger:
    """Centralized debug logger that respects framework configuration."""
    
    def __init__(self, enabled: Optional[bool] = None):
        """Initialize debug logger.
        
        Args:
            enabled: If None, reads from config. If True/False, overrides config.
        """
        self._enabled = enabled
        self._settings = None
    
    def _get_enabled(self) -> bool:
        """Get whether debug is enabled."""
        if self._enabled is not None:
            return self._enabled
        if self._settings is None:
            self._settings = get_settings()
        return self._settings.DEBUG
    
    def log(self, message: str, category: Optional[str] = None) -> None:
        """Log a debug message.
        
        Args:
            message: The debug message to log.
            category: Optional category prefix (e.g., "agent", "rag", "tool").
        """
        if not self._get_enabled():
            return
        
        prefix = "[DEBUG]"
        if category:
            prefix = f"[DEBUG:{category.upper()}]"
        
        print(f"{prefix} {message}", file=sys.stderr)
    
    def log_tool_call(self, tool_name: str, args: dict) -> None:
        """Log a tool call.
        
        Args:
            tool_name: Name of the tool being called.
            args: Arguments passed to the tool.
        """
        self.log(f"Tool call: {tool_name}({args})", category="tool")
    
    def log_message(self, msg_type: str, content_preview: str, index: Optional[int] = None) -> None:
        """Log a message (e.g., AIMessage, HumanMessage).
        
        Args:
            msg_type: Type of message (e.g., "AIMessage", "HumanMessage").
            content_preview: Preview of message content.
            index: Optional message index.
        """
        prefix = f"Message {index}" if index is not None else "Message"
        self.log(f"{prefix}: {msg_type} - {content_preview}...", category="message")
    
    def log_warning(self, message: str) -> None:
        """Log a warning message.
        
        Args:
            message: Warning message.
        """
        self.log(f"⚠️  WARNING: {message}", category="warning")


# Global debug logger instance
_global_logger: Optional[DebugLogger] = None


def get_debug_logger(enabled: Optional[bool] = None) -> DebugLogger:
    """Get the global debug logger instance.
    
    Args:
        enabled: Optional override for debug enabled state.
        
    Returns:
        DebugLogger instance.
    """
    global _global_logger
    if _global_logger is None or enabled is not None:
        _global_logger = DebugLogger(enabled=enabled)
    return _global_logger


def debug_log(message: str, category: Optional[str] = None, enabled: Optional[bool] = None) -> None:
    """Convenience function to log a debug message.
    
    Args:
        message: The debug message to log.
        category: Optional category prefix.
        enabled: Optional override for debug enabled state.
    """
    logger = get_debug_logger(enabled=enabled)
    logger.log(message, category=category)


def is_debug_enabled(enabled: Optional[bool] = None) -> bool:
    """Check if debug logging is enabled.
    
    Args:
        enabled: Optional override for debug enabled state.
        
    Returns:
        True if debug is enabled, False otherwise.
    """
    logger = get_debug_logger(enabled=enabled)
    return logger._get_enabled()


def set_debug_enabled(enabled: bool) -> None:
    """Set debug enabled state globally.
    
    Args:
        enabled: Whether debug should be enabled.
    """
    global _global_logger
    _global_logger = DebugLogger(enabled=enabled)
