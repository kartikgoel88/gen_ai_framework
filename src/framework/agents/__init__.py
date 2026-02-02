"""Agents: ReAct and tool-calling agents using framework LLM and tools."""

"""Agent framework with advanced agentic AI capabilities.

This module provides:
- Base agent interface
- ReAct agents with tool support
- Streaming agents with intermediate step visibility
- Agents with memory and persistence
- Multi-agent systems
- Planning agents
- Reflective agents with self-correction
- Monitoring and cost tracking
- Persona-based agents
- Error recovery and retry logic
"""

from .base import AgentBase
from .langchain_agent import LangChainReActAgent
from .streaming_agent import StreamingAgent, AgentEvent, AgentEventType
from .memory import MemoryStore, RAGMemoryStore, AgentWithMemory, ConversationMemory
from .multi_agent import (
    MultiAgentSystem,
    ResearcherAgent,
    WriterAgent,
    ReviewerAgent,
    AgentRole
)
from .planning_agent import PlanningAgent, ExecutionPlan, PlanStep, PlanStatus
from .reflective_agent import ReflectiveAgent, Reflection, ReflectionResult
from .monitoring import AgentMonitor, MonitoredAgent, MonitoringEvent, EventType, AgentMetrics
from .cost_tracking import (
    CostTracker,
    CostTrackingAgent,
    CostEntry,
    BudgetExceededError
)
from .personas import (
    Persona,
    PersonaAgent,
    PersonaType,
    PERSONAS,
    create_persona_agent,
    create_custom_persona
)
from .error_recovery import (
    ErrorRecoveryAgent,
    RetryConfig,
    ErrorType,
    create_simple_fallback
)
from .tools import build_framework_tools

__all__ = [
    # Base
    "AgentBase",
    "LangChainReActAgent",
    # Streaming
    "StreamingAgent",
    "AgentEvent",
    "AgentEventType",
    # Memory
    "MemoryStore",
    "RAGMemoryStore",
    "AgentWithMemory",
    "ConversationMemory",
    # Multi-Agent
    "MultiAgentSystem",
    "ResearcherAgent",
    "WriterAgent",
    "ReviewerAgent",
    "AgentRole",
    # Planning
    "PlanningAgent",
    "ExecutionPlan",
    "PlanStep",
    "PlanStatus",
    # Reflection
    "ReflectiveAgent",
    "Reflection",
    "ReflectionResult",
    # Monitoring
    "AgentMonitor",
    "MonitoredAgent",
    "MonitoringEvent",
    "EventType",
    "AgentMetrics",
    # Cost Tracking
    "CostTracker",
    "CostTrackingAgent",
    "CostEntry",
    "BudgetExceededError",
    # Personas
    "Persona",
    "PersonaAgent",
    "PersonaType",
    "PERSONAS",
    "create_persona_agent",
    "create_custom_persona",
    # Error Recovery
    "ErrorRecoveryAgent",
    "RetryConfig",
    "ErrorType",
    "create_simple_fallback",
    # Tools
    "build_framework_tools",
]
