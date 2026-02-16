"""Agent planning capabilities for complex multi-step tasks."""

from typing import Any, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum

from .base import AgentBase
from .planning_prompts import (
    format_plan_create,
    format_step_execute,
    format_plan_revise,
    format_result_compile,
)


class PlanStatus(Enum):
    """Status of a plan step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """Represents a step in an execution plan."""
    step_number: int
    description: str
    status: PlanStatus
    result: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ExecutionPlan:
    """Represents an execution plan."""
    goal: str
    steps: List[PlanStep]
    status: PlanStatus = PlanStatus.PENDING
    
    def get_next_step(self) -> Optional[PlanStep]:
        """Get the next pending step."""
        for step in self.steps:
            if step.status == PlanStatus.PENDING:
                return step
        return None
    
    def is_complete(self) -> bool:
        """Check if plan is complete."""
        return all(step.status in [PlanStatus.COMPLETED, PlanStatus.SKIPPED] for step in self.steps)
    
    def has_failures(self) -> bool:
        """Check if plan has failures."""
        return any(step.status == PlanStatus.FAILED for step in self.steps)


class PlanningAgent(AgentBase):
    """Agent that plans before acting (Plan-and-Solve pattern).
    
    This agent creates a step-by-step plan before execution,
    then executes each step, monitoring progress and revising
    the plan if needed.
    """
    
    def __init__(
        self,
        base_agent: AgentBase,
        enable_revision: bool = True,
        max_revisions: int = 3
    ):
        """Initialize planning agent.
        
        Args:
            base_agent: Base agent to use for execution
            enable_revision: Enable plan revision on failure
            max_revisions: Maximum number of plan revisions
        """
        self._base_agent = base_agent
        self._enable_revision = enable_revision
        self._max_revisions = max_revisions
    
    def invoke(self, message: str, **kwargs) -> str:
        """Invoke agent with planning.
        
        Args:
            message: Task description
            **kwargs: Additional arguments
            
        Returns:
            Final result
        """
        # Generate initial plan
        plan = self._create_plan(message)
        
        revision_count = 0
        
        while not plan.is_complete() and revision_count < self._max_revisions:
            # Execute plan
            plan = self._execute_plan(plan, **kwargs)
            
            # Revise plan if needed
            if plan.has_failures() and self._enable_revision:
                plan = self._revise_plan(plan, message)
                revision_count += 1
        
        # Compile final result
        return self._compile_result(plan)
    
    def _create_plan(self, goal: str) -> ExecutionPlan:
        """Create an execution plan for the goal."""
        plan_text = self._base_agent.invoke(format_plan_create(goal))
        
        # Parse plan into steps
        steps = self._parse_plan(plan_text)
        
        return ExecutionPlan(
            goal=goal,
            steps=steps
        )
    
    def _parse_plan(self, plan_text: str) -> List[PlanStep]:
        """Parse plan text into PlanStep objects."""
        steps = []
        lines = plan_text.split("\n")
        step_num = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts with a number
            if line[0].isdigit():
                # Extract step description
                desc = line.split(".", 1)[-1].strip()
                if desc:
                    steps.append(PlanStep(
                        step_number=step_num,
                        description=desc,
                        status=PlanStatus.PENDING
                    ))
                    step_num += 1
        
        return steps
    
    def _execute_plan(self, plan: ExecutionPlan, **kwargs) -> ExecutionPlan:
        """Execute the plan step by step."""
        plan.status = PlanStatus.IN_PROGRESS
        
        for step in plan.steps:
            if step.status != PlanStatus.PENDING:
                continue
            
            step.status = PlanStatus.IN_PROGRESS
            
            try:
                previous = [s.description for s in plan.steps if s.status == PlanStatus.COMPLETED]
                step_prompt = format_step_execute(step.description, plan.goal, previous)
                result = self._base_agent.invoke(step_prompt, **kwargs)
                step.result = result
                step.status = PlanStatus.COMPLETED
                
            except Exception as e:
                step.status = PlanStatus.FAILED
                step.error = str(e)
                # Stop execution on failure (can be made configurable)
                break
        
        if plan.is_complete():
            plan.status = PlanStatus.COMPLETED
        
        return plan
    
    def _revise_plan(self, plan: ExecutionPlan, original_goal: str) -> ExecutionPlan:
        """Revise plan based on failures."""
        failed_steps = [s for s in plan.steps if s.status == PlanStatus.FAILED]
        failed_lines = [
            f"{s.step_number}. {s.description} - Error: {s.error}" for s in failed_steps
        ]
        completed_lines = [
            f"{s.step_number}. {s.description}"
            for s in plan.steps
            if s.status == PlanStatus.COMPLETED
        ]
        revision_prompt = format_plan_revise(
            original_goal,
            "\n".join(failed_lines),
            "\n".join(completed_lines),
        )
        revised_plan_text = self._base_agent.invoke(revision_prompt)
        
        # Parse revised plan
        new_steps = self._parse_plan(revised_plan_text)
        
        # Keep completed steps, replace failed ones
        completed_steps = [s for s in plan.steps if s.status == PlanStatus.COMPLETED]
        
        plan.steps = completed_steps + new_steps
        plan.status = PlanStatus.PENDING
        
        return plan
    
    def _compile_result(self, plan: ExecutionPlan) -> str:
        """Compile final result from plan execution."""
        if plan.status == PlanStatus.COMPLETED:
            completed_parts = [
                f"Step {s.step_number}: {s.description}\nResult: {s.result or '(none)'}"
                for s in plan.steps
                if s.status == PlanStatus.COMPLETED
            ]
            compilation_prompt = format_result_compile(
                plan.goal,
                "\n\n".join(completed_parts),
            )
            return self._base_agent.invoke(compilation_prompt)
        else:
            # Return partial results
            results = []
            for step in plan.steps:
                if step.status == PlanStatus.COMPLETED:
                    results.append(f"Step {step.step_number}: {step.result}")
            
            return "\n\n".join(results) if results else "Plan execution incomplete."
    
    def get_tools_description(self) -> list[dict[str, Any]]:
        return self._base_agent.get_tools_description()
