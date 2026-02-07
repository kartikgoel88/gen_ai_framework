"""Error recovery and retry logic for agents."""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional

from ..exceptions import FrameworkError
from ..utils.retry_utils import compute_backoff_delay, is_rate_limit_error
from .base import AgentBase


class ErrorType(Enum):
    """Types of errors."""
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    INVALID_INPUT = "invalid_input"
    TOOL_ERROR = "tool_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    backoff_factor: float = 2.0
    initial_delay: float = 1.0
    max_delay: float = 60.0
    retryable_errors: List[ErrorType] = None
    
    def __post_init__(self):
        if self.retryable_errors is None:
            self.retryable_errors = [
                ErrorType.NETWORK,
                ErrorType.RATE_LIMIT,
                ErrorType.TIMEOUT
            ]


class ErrorRecoveryAgent(AgentBase):
    """Agent with automatic error recovery and retry logic."""
    
    def __init__(
        self,
        base_agent: AgentBase,
        retry_config: Optional[RetryConfig] = None,
        fallback_strategy: Optional[Callable] = None
    ):
        """Initialize error recovery agent.
        
        Args:
            base_agent: Base agent to wrap
            retry_config: Retry configuration
            fallback_strategy: Optional fallback function
        """
        self._base_agent = base_agent
        self._retry_config = retry_config or RetryConfig()
        self._fallback_strategy = fallback_strategy
    
    def invoke(self, message: str, **kwargs) -> str:
        """Invoke agent with error recovery.
        
        Args:
            message: User message
            **kwargs: Additional arguments
            
        Returns:
            Agent response
            
        Raises:
            FrameworkError: If all retries fail
        """
        last_error = None
        
        for attempt in range(self._retry_config.max_retries + 1):
            try:
                return self._base_agent.invoke(message, **kwargs)
                
            except Exception as e:
                last_error = e
                error_type = self._classify_error(e)
                
                # Check if error is retryable
                if error_type not in self._retry_config.retryable_errors:
                    # Not retryable, try fallback or raise
                    if self._fallback_strategy:
                        return self._fallback_strategy(message, e, **kwargs)
                    raise
                
                # Check if we have retries left
                if attempt < self._retry_config.max_retries:
                    delay = compute_backoff_delay(
                        attempt,
                        initial_delay=self._retry_config.initial_delay,
                        factor=self._retry_config.backoff_factor,
                        max_delay=self._retry_config.max_delay,
                        rate_limit_min=60.0 if error_type == ErrorType.RATE_LIMIT else None,
                    )
                    time.sleep(delay)
                    continue
                else:
                    # Out of retries, try fallback or raise
                    if self._fallback_strategy:
                        return self._fallback_strategy(message, e, **kwargs)
                    raise FrameworkError(
                        f"Agent invocation failed after {self._retry_config.max_retries} retries: {str(e)}"
                    ) from e
        
        # Should not reach here, but just in case
        if self._fallback_strategy:
            return self._fallback_strategy(message, last_error, **kwargs)
        raise FrameworkError(f"Agent invocation failed: {str(last_error)}") from last_error
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error type.
        
        Args:
            error: Exception to classify
            
        Returns:
            ErrorType
        """
        error_str = str(error).lower()
        if is_rate_limit_error(error):
            return ErrorType.RATE_LIMIT
        elif "timeout" in error_str or "timed out" in error_str:
            return ErrorType.TIMEOUT
        elif "network" in error_str or "connection" in error_str:
            return ErrorType.NETWORK
        elif "invalid" in error_str or "validation" in error_str:
            return ErrorType.INVALID_INPUT
        elif "tool" in error_str:
            return ErrorType.TOOL_ERROR
        else:
            return ErrorType.UNKNOWN
    
    def get_tools_description(self) -> list[dict[str, Any]]:
        return self._base_agent.get_tools_description()


def create_simple_fallback(message: str, error: Exception, **kwargs) -> str:
    """Simple fallback strategy that returns an error message.
    
    Args:
        message: Original message
        error: Error that occurred
        **kwargs: Additional arguments
        
    Returns:
        Fallback response
    """
    return f"I apologize, but I encountered an error while processing your request: {str(error)}. Please try again later or rephrase your question."
