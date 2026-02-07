"""Error recovery handlers for agent operations."""

import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from ..utils.retry_utils import compute_backoff_delay, is_rate_limit_error
from .errors import ToolCallError, ToolCallParseError, InvalidToolArgumentsError


class ErrorRecoveryHandler(ABC):
    """Abstract base class for error recovery strategies."""
    
    @abstractmethod
    def can_handle(self, error: Exception) -> bool:
        """Check if this handler can handle the given error."""
        ...
    
    @abstractmethod
    def handle(self, error: Exception, context: dict[str, Any]) -> Any:
        """
        Handle the error and return recovery action.
        
        Args:
            error: The exception that occurred
            context: Additional context (e.g., tool_name, arguments, retry_count)
            
        Returns:
            Recovery result or raises if cannot recover
        """
        ...


class ToolCallRetryHandler(ErrorRecoveryHandler):
    """Retry handler for tool call parsing errors."""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
    
    def can_handle(self, error: Exception) -> bool:
        """Handle tool call parse errors."""
        return isinstance(error, ToolCallParseError)
    
    def handle(self, error: Exception, context: dict[str, Any]) -> Any:
        """Retry parsing with different strategies."""
        retry_count = context.get("retry_count", 0)
        
        if retry_count >= self.max_retries:
            raise error
        
        # Try alternative parsing strategies
        raw_response = getattr(error, "raw_response", "")
        if raw_response:
            # Try extracting JSON more aggressively
            import json
            import re
            
            # Try to find any JSON-like structure
            json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_response)
            for match in json_matches:
                try:
                    data = json.loads(match)
                    if "name" in data and "arguments" in data:
                        return data
                except json.JSONDecodeError:
                    continue
        
        # If all retries exhausted, re-raise
        raise error


class ParseErrorHandler(ErrorRecoveryHandler):
    """Handler for general parsing errors."""
    
    def can_handle(self, error: Exception) -> bool:
        """Handle JSON decode errors and similar."""
        return isinstance(error, (ValueError, KeyError)) or "parse" in str(error).lower()
    
    def handle(self, error: Exception, context: dict[str, Any]) -> Any:
        """Return empty result or fallback."""
        # Return empty tool calls list as fallback
        return []


class RateLimitHandler(ErrorRecoveryHandler):
    """Handler for rate limit errors."""
    
    def __init__(self, backoff_factor: float = 2.0, max_wait: float = 60.0):
        self.backoff_factor = backoff_factor
        self.max_wait = max_wait
    
    def can_handle(self, error: Exception) -> bool:
        """Handle rate limit errors."""
        return is_rate_limit_error(error)
    
    def handle(self, error: Exception, context: dict[str, Any]) -> Any:
        """Wait and retry with exponential backoff."""
        retry_count = context.get("retry_count", 0)
        wait_time = compute_backoff_delay(
            retry_count,
            initial_delay=1.0,
            factor=self.backoff_factor,
            max_delay=self.max_wait,
            rate_limit_min=60.0,
        )
        time.sleep(wait_time)
        raise error  # Re-raise to trigger retry


class ErrorRecoveryManager:
    """Manages multiple error recovery handlers."""
    
    def __init__(self):
        self.handlers: list[ErrorRecoveryHandler] = [
            ToolCallRetryHandler(),
            ParseErrorHandler(),
            RateLimitHandler(),
        ]
    
    def add_handler(self, handler: ErrorRecoveryHandler):
        """Add a custom error recovery handler."""
        self.handlers.append(handler)
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[dict[str, Any]] = None
    ) -> Any:
        """
        Attempt to recover from an error using registered handlers.
        
        Args:
            error: The exception to handle
            context: Additional context for recovery
            
        Returns:
            Recovery result
            
        Raises:
            The original error if no handler can recover
        """
        context = context or {}
        
        for handler in self.handlers:
            if handler.can_handle(error):
                try:
                    return handler.handle(error, context)
                except Exception as recovery_error:
                    # If recovery fails, try next handler
                    if recovery_error is error:
                        raise  # Re-raise original if handler explicitly raises it
                    continue
        
        # No handler could recover
        raise error
