"""Event system for agent observability."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentEventType(Enum):
    """Types of agent events."""
    TOOL_SELECTION = "tool_selection"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    ERROR = "error"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"


@dataclass
class AgentEvent:
    """Represents an agent event."""
    
    event_type: AgentEventType
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            import time
            self.timestamp = time.time()


class AgentObserver(ABC):
    """Abstract observer for agent events."""
    
    @abstractmethod
    def on_event(self, event: AgentEvent) -> None:
        """Handle an agent event."""
        ...


class DebugObserver(AgentObserver):
    """Observer that logs events for debugging."""
    
    def __init__(self, logger=None):
        """
        Initialize with optional logger.
        
        Args:
            logger: Optional logger instance (defaults to print)
        """
        self.logger = logger
    
    def on_event(self, event: AgentEvent) -> None:
        """Log event."""
        message = f"[{event.event_type.value}] {event.content}"
        if event.metadata:
            message += f" | Metadata: {event.metadata}"
        
        if self.logger:
            self.logger.log(message, category="agent")
        else:
            print(f"[DEBUG:AGENT] {message}")


class MetricsObserver(AgentObserver):
    """Observer that collects metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {
            "tool_calls": 0,
            "errors": 0,
            "messages": 0,
        }
    
    def on_event(self, event: AgentEvent) -> None:
        """Update metrics."""
        if event.event_type == AgentEventType.TOOL_CALL:
            self.metrics["tool_calls"] += 1
        elif event.event_type == AgentEventType.ERROR:
            self.metrics["errors"] += 1
        elif event.event_type in (AgentEventType.MESSAGE_SENT, AgentEventType.MESSAGE_RECEIVED):
            self.metrics["messages"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        return self.metrics.copy()
    
    def reset(self) -> None:
        """Reset metrics."""
        self.metrics = {
            "tool_calls": 0,
            "errors": 0,
            "messages": 0,
        }


class EventEmitter:
    """Manages observers and emits events."""
    
    def __init__(self):
        self._observers: List[AgentObserver] = []
    
    def subscribe(self, observer: AgentObserver) -> None:
        """Subscribe an observer."""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def unsubscribe(self, observer: AgentObserver) -> None:
        """Unsubscribe an observer."""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def emit(self, event: AgentEvent) -> None:
        """Emit an event to all observers."""
        for observer in self._observers:
            try:
                observer.on_event(event)
            except Exception:
                # Don't let observer errors break the agent
                pass
    
    def clear(self) -> None:
        """Clear all observers."""
        self._observers.clear()
