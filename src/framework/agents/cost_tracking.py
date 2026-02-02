"""Cost tracking and budget management for agents."""

from typing import Any, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from .base import AgentBase
from ..exceptions import FrameworkError


class BudgetExceededError(FrameworkError):
    """Raised when budget limit is exceeded."""
    pass


@dataclass
class CostEntry:
    """Represents a cost entry."""
    timestamp: datetime
    agent_id: str
    operation: str
    input_tokens: int
    output_tokens: int
    cost: float
    model: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class CostTracker:
    """Tracks costs for agent operations."""
    
    # Cost per 1K tokens (approximate, update as needed)
    MODEL_COSTS = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "grok-2": {"input": 0.01, "output": 0.03},
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    }
    
    def __init__(self):
        """Initialize cost tracker."""
        self._entries: list[CostEntry] = []
        self._budgets: Dict[str, float] = {}  # agent_id -> budget
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost for token usage.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        costs = self.MODEL_COSTS.get(model.lower(), {"input": 0.002, "output": 0.002})
        
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        
        return input_cost + output_cost
    
    def track(
        self,
        agent_id: str,
        operation: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """Track a cost entry.
        
        Args:
            agent_id: Agent identifier
            operation: Operation name
            model: Model used
            input_tokens: Input tokens
            output_tokens: Output tokens
            metadata: Additional metadata
            
        Returns:
            Calculated cost
            
        Raises:
            BudgetExceededError: If budget is exceeded
        """
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        
        entry = CostEntry(
            timestamp=datetime.now(),
            agent_id=agent_id,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            model=model,
            metadata=metadata or {}
        )
        
        self._entries.append(entry)
        
        # Check budget
        total_cost = self.get_total_cost(agent_id)
        budget = self._budgets.get(agent_id)
        
        if budget and total_cost > budget:
            raise BudgetExceededError(
                f"Budget exceeded for agent {agent_id}. "
                f"Total: ${total_cost:.4f}, Budget: ${budget:.4f}"
            )
        
        return cost
    
    def set_budget(self, agent_id: str, budget: float) -> None:
        """Set budget for an agent.
        
        Args:
            agent_id: Agent identifier
            budget: Budget in USD
        """
        self._budgets[agent_id] = budget
    
    def get_total_cost(self, agent_id: Optional[str] = None) -> float:
        """Get total cost for an agent or all agents.
        
        Args:
            agent_id: Optional agent identifier
            
        Returns:
            Total cost in USD
        """
        if agent_id:
            return sum(e.cost for e in self._entries if e.agent_id == agent_id)
        return sum(e.cost for e in self._entries)
    
    def get_cost_summary(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get cost summary.
        
        Args:
            agent_id: Optional agent identifier
            
        Returns:
            Cost summary dictionary
        """
        entries = [e for e in self._entries if not agent_id or e.agent_id == agent_id]
        
        if not entries:
            return {
                "total_cost": 0.0,
                "total_operations": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "by_model": {}
            }
        
        by_model = defaultdict(lambda: {"cost": 0.0, "operations": 0, "tokens": 0})
        
        for entry in entries:
            by_model[entry.model]["cost"] += entry.cost
            by_model[entry.model]["operations"] += 1
            by_model[entry.model]["tokens"] += entry.input_tokens + entry.output_tokens
        
        return {
            "total_cost": sum(e.cost for e in entries),
            "total_operations": len(entries),
            "total_input_tokens": sum(e.input_tokens for e in entries),
            "total_output_tokens": sum(e.output_tokens for e in entries),
            "by_model": dict(by_model)
        }
    
    def get_entries(self, agent_id: Optional[str] = None) -> list[CostEntry]:
        """Get cost entries."""
        if agent_id:
            return [e for e in self._entries if e.agent_id == agent_id]
        return self._entries.copy()
    
    def clear(self, agent_id: Optional[str] = None) -> None:
        """Clear cost entries."""
        if agent_id:
            self._entries = [e for e in self._entries if e.agent_id != agent_id]
        else:
            self._entries.clear()


class CostTrackingAgent(AgentBase):
    """Wrapper that adds cost tracking to any agent."""
    
    def __init__(
        self,
        base_agent: AgentBase,
        cost_tracker: CostTracker,
        agent_id: str = "default",
        model: str = "gpt-4-turbo-preview"
    ):
        """Initialize cost tracking agent.
        
        Args:
            base_agent: Base agent to track
            cost_tracker: Cost tracker instance
            agent_id: Agent identifier
            model: Model name for cost calculation
        """
        self._base_agent = base_agent
        self._cost_tracker = cost_tracker
        self._agent_id = agent_id
        self._model = model
    
    def invoke(self, message: str, **kwargs) -> str:
        """Invoke agent with cost tracking."""
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        input_tokens = len(message) // 4
        
        response = self._base_agent.invoke(message, **kwargs)
        
        output_tokens = len(response) // 4
        
        # Track cost
        self._cost_tracker.track(
            agent_id=self._agent_id,
            operation="invoke",
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={"message_length": len(message), "response_length": len(response)}
        )
        
        return response
    
    def get_tools_description(self) -> list[dict[str, Any]]:
        return self._base_agent.get_tools_description()
