"""Multi-agent systems for complex task orchestration."""

from typing import Any, Optional, List, Dict, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .base import AgentBase


@dataclass
class AgentRole:
    """Defines an agent's role and capabilities."""
    name: str
    description: str
    system_prompt: str
    allowed_tools: List[str]
    capabilities: List[str]


class MultiAgentSystem:
    """Orchestrates multiple specialized agents working together.
    
    Example:
        ```python
        researcher = ResearcherAgent(...)
        writer = WriterAgent(...)
        reviewer = ReviewerAgent(...)
        
        system = MultiAgentSystem(
            agents={
                "researcher": researcher,
                "writer": writer,
                "reviewer": reviewer
            }
        )
        
        result = system.invoke("Write a comprehensive guide on RAG")
        ```
    """
    
    def __init__(
        self,
        agents: Dict[str, AgentBase],
        orchestrator: Optional[AgentBase] = None,
        workflow: Optional[Callable] = None
    ):
        """Initialize multi-agent system.
        
        Args:
            agents: Dictionary of agent name -> agent instance
            orchestrator: Optional orchestrator agent for task routing
            workflow: Optional custom workflow function
        """
        self._agents = agents
        self._orchestrator = orchestrator
        self._workflow = workflow or self._default_workflow
    
    def invoke(self, task: str, **kwargs) -> str:
        """Invoke multi-agent system on a task.
        
        Args:
            task: Task description
            **kwargs: Additional arguments
            
        Returns:
            Final result from the workflow
        """
        return self._workflow(task, **kwargs)
    
    def _default_workflow(self, task: str, **kwargs) -> str:
        """Default sequential workflow.
        
        Executes agents in order, passing results between them.
        """
        result = task
        
        # Execute agents sequentially
        for name, agent in self._agents.items():
            if self._orchestrator:
                # Use orchestrator to decide if agent should run
                decision = self._orchestrator.invoke(
                    f"Should {name} process this task? Task: {result}"
                )
                if "no" in decision.lower() or "skip" in decision.lower():
                    continue
            
            result = agent.invoke(result, **kwargs)
        
        return result
    
    def get_agents(self) -> Dict[str, AgentBase]:
        """Get all agents in the system."""
        return self._agents
    
    def add_agent(self, name: str, agent: AgentBase) -> None:
        """Add an agent to the system."""
        self._agents[name] = agent
    
    def remove_agent(self, name: str) -> None:
        """Remove an agent from the system."""
        if name in self._agents:
            del self._agents[name]


class ResearcherAgent(AgentBase):
    """Specialized agent for research tasks."""
    
    def __init__(self, base_agent: AgentBase):
        """Initialize researcher agent."""
        self._base_agent = base_agent
    
    def invoke(self, message: str, **kwargs) -> str:
        """Conduct research on the topic."""
        research_prompt = f"""You are a research specialist. Conduct thorough research on the following topic:
        
{message}

Provide comprehensive information, facts, and sources."""
        
        return self._base_agent.invoke(research_prompt, **kwargs)
    
    def get_tools_description(self) -> list[dict[str, Any]]:
        return self._base_agent.get_tools_description()


class WriterAgent(AgentBase):
    """Specialized agent for writing tasks."""
    
    def __init__(self, base_agent: AgentBase):
        """Initialize writer agent."""
        self._base_agent = base_agent
    
    def invoke(self, message: str, **kwargs) -> str:
        """Write content based on the input."""
        writing_prompt = f"""You are a technical writer. Write clear, well-structured content based on the following information:
        
{message}

Ensure the content is well-organized, accurate, and easy to understand."""
        
        return self._base_agent.invoke(writing_prompt, **kwargs)
    
    def get_tools_description(self) -> list[dict[str, Any]]:
        return self._base_agent.get_tools_description()


class ReviewerAgent(AgentBase):
    """Specialized agent for review and quality assurance."""
    
    def __init__(self, base_agent: AgentBase):
        """Initialize reviewer agent."""
        self._base_agent = base_agent
    
    def invoke(self, message: str, **kwargs) -> str:
        """Review and improve the content."""
        review_prompt = f"""You are a quality reviewer. Review the following content and provide improvements:
        
{message}

Check for accuracy, clarity, completeness, and suggest improvements."""
        
        return self._base_agent.invoke(review_prompt, **kwargs)
    
    def get_tools_description(self) -> list[dict[str, Any]]:
        return self._base_agent.get_tools_description()
