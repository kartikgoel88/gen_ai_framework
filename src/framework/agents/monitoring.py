"""Agent monitoring and observability."""

from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

from .base import AgentBase


class EventType(Enum):
    """Types of monitoring events."""
    INVOCATION_START = "invocation_start"
    INVOCATION_END = "invocation_end"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    COST = "cost"


@dataclass
class MonitoringEvent:
    """Represents a monitoring event."""
    event_type: EventType
    timestamp: datetime
    agent_id: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMetrics:
    """Aggregated metrics for an agent."""
    total_invocations: int = 0
    total_tool_calls: int = 0
    total_errors: int = 0
    total_cost: float = 0.0
    average_latency: float = 0.0
    tool_usage: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_rate: float = 0.0


class AgentMonitor:
    """Monitors agent execution and collects metrics."""
    
    def __init__(self):
        """Initialize monitor."""
        self._events: List[MonitoringEvent] = []
        self._metrics: Dict[str, AgentMetrics] = {}
    
    def track_event(
        self,
        event_type: EventType,
        agent_id: str,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track a monitoring event."""
        event = MonitoringEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            agent_id=agent_id,
            data=data or {}
        )
        self._events.append(event)
        self._update_metrics(agent_id, event)
    
    def _update_metrics(self, agent_id: str, event: MonitoringEvent) -> None:
        """Update metrics based on event."""
        if agent_id not in self._metrics:
            self._metrics[agent_id] = AgentMetrics()
        
        metrics = self._metrics[agent_id]
        
        if event.event_type == EventType.INVOCATION_START:
            metrics.total_invocations += 1
        elif event.event_type == EventType.TOOL_CALL:
            metrics.total_tool_calls += 1
            tool_name = event.data.get("tool_name", "unknown")
            metrics.tool_usage[tool_name] += 1
        elif event.event_type == EventType.ERROR:
            metrics.total_errors += 1
        elif event.event_type == EventType.COST:
            metrics.total_cost += event.data.get("cost", 0.0)
        
        # Update error rate
        if metrics.total_invocations > 0:
            metrics.error_rate = metrics.total_errors / metrics.total_invocations
    
    def get_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """Get metrics for an agent."""
        return self._metrics.get(agent_id)
    
    def get_all_metrics(self) -> Dict[str, AgentMetrics]:
        """Get all agent metrics."""
        return self._metrics.copy()
    
    def get_events(self, agent_id: Optional[str] = None) -> List[MonitoringEvent]:
        """Get events, optionally filtered by agent."""
        if agent_id:
            return [e for e in self._events if e.agent_id == agent_id]
        return self._events.copy()
    
    def clear(self, agent_id: Optional[str] = None) -> None:
        """Clear events and metrics."""
        if agent_id:
            self._events = [e for e in self._events if e.agent_id != agent_id]
            if agent_id in self._metrics:
                del self._metrics[agent_id]
        else:
            self._events.clear()
            self._metrics.clear()


class MonitoredAgent(AgentBase):
    """Wrapper that adds monitoring to any agent."""
    
    def __init__(self, base_agent: AgentBase, monitor: AgentMonitor, agent_id: str = "default"):
        """Initialize monitored agent.
        
        Args:
            base_agent: Base agent to monitor
            monitor: Monitor instance
            agent_id: Unique identifier for this agent
        """
        self._base_agent = base_agent
        self._monitor = monitor
        self._agent_id = agent_id
    
    def invoke(self, message: str, **kwargs) -> str:
        """Invoke agent with monitoring."""
        import time
        
        start_time = time.time()
        
        self._monitor.track_event(
            EventType.INVOCATION_START,
            self._agent_id,
            {"message": message[:100]}  # Truncate for storage
        )
        
        try:
            response = self._base_agent.invoke(message, **kwargs)
            
            latency = time.time() - start_time
            
            self._monitor.track_event(
                EventType.INVOCATION_END,
                self._agent_id,
                {"latency": latency, "response_length": len(response)}
            )
            
            # Update average latency
            metrics = self._monitor.get_metrics(self._agent_id)
            if metrics:
                total = metrics.total_invocations
                if total > 0:
                    metrics.average_latency = (
                        (metrics.average_latency * (total - 1) + latency) / total
                    )
            
            return response
            
        except Exception as e:
            self._monitor.track_event(
                EventType.ERROR,
                self._agent_id,
                {"error": str(e), "error_type": type(e).__name__}
            )
            raise
    
    def get_tools_description(self) -> list[dict[str, Any]]:
        return self._base_agent.get_tools_description()
