"""Agent evaluation framework."""

from typing import Any, List, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from .base import AgentBase


class MetricType(Enum):
    """Types of evaluation metrics."""
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    LATENCY = "latency"
    COST = "cost"
    TOOL_USAGE = "tool_usage"
    CUSTOM = "custom"


@dataclass
class EvaluationTask:
    """Represents an evaluation task."""
    task_id: str
    prompt: str
    expected_output: Optional[str] = None
    expected_tools: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Represents an evaluation result."""
    task_id: str
    actual_output: str
    metrics: Dict[str, float] = field(default_factory=dict)
    tool_usage: List[str] = field(default_factory=list)
    latency: float = 0.0
    cost: float = 0.0
    passed: bool = False
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Metric(ABC):
    """Abstract base class for evaluation metrics."""
    
    @abstractmethod
    def compute(
        self,
        task: EvaluationTask,
        result: EvaluationResult
    ) -> float:
        """Compute metric value.
        
        Args:
            task: Evaluation task
            result: Evaluation result
            
        Returns:
            Metric value (typically 0.0 to 1.0)
        """
        ...


class ExactMatchMetric(Metric):
    """Exact match metric."""
    
    def compute(self, task: EvaluationTask, result: EvaluationResult) -> float:
        """Compute exact match score."""
        if not task.expected_output:
            return 0.0
        
        expected = task.expected_output.strip().lower()
        actual = result.actual_output.strip().lower()
        
        return 1.0 if expected == actual else 0.0


class KeywordMatchMetric(Metric):
    """Keyword match metric."""
    
    def compute(self, task: EvaluationTask, result: EvaluationResult) -> float:
        """Compute keyword match score."""
        if not task.expected_output:
            return 0.0
        
        expected_keywords = set(task.expected_output.lower().split())
        actual_keywords = set(result.actual_output.lower().split())
        
        if not expected_keywords:
            return 0.0
        
        intersection = expected_keywords & actual_keywords
        return len(intersection) / len(expected_keywords)


class ToolUsageMetric(Metric):
    """Tool usage metric."""
    
    def compute(self, task: EvaluationTask, result: EvaluationResult) -> float:
        """Compute tool usage score."""
        if not task.expected_tools:
            return 1.0  # No expected tools means tool usage is correct
        
        expected_set = set(task.expected_tools)
        actual_set = set(result.tool_usage)
        
        # Check if expected tools were used
        used_expected = len(expected_set & actual_set)
        return used_expected / len(expected_set) if expected_set else 1.0


class AgentEvaluator:
    """Evaluates agent performance on tasks."""
    
    def __init__(
        self,
        metrics: Optional[List[Metric]] = None,
        custom_scorer: Optional[Callable] = None
    ):
        """Initialize evaluator.
        
        Args:
            metrics: List of metrics to compute
            custom_scorer: Optional custom scoring function
        """
        self._metrics = metrics or [
            ExactMatchMetric(),
            KeywordMatchMetric(),
            ToolUsageMetric()
        ]
        self._custom_scorer = custom_scorer
    
    def evaluate(
        self,
        agent: AgentBase,
        tasks: List[EvaluationTask],
        track_tools: bool = True,
        track_latency: bool = True
    ) -> List[EvaluationResult]:
        """Evaluate agent on tasks.
        
        Args:
            agent: Agent to evaluate
            tasks: List of evaluation tasks
            track_tools: Track tool usage
            track_latency: Track latency
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for task in tasks:
            import time
            
            start_time = time.time()
            
            try:
                # Invoke agent
                output = agent.invoke(task.prompt, **(task.context or {}))
                
                latency = time.time() - start_time if track_latency else 0.0
                
                # Get tool usage (if available)
                tool_usage = []
                if track_tools and hasattr(agent, "_tools_used"):
                    tool_usage = agent._tools_used
                
                # Create result
                result = EvaluationResult(
                    task_id=task.task_id,
                    actual_output=output,
                    tool_usage=tool_usage,
                    latency=latency
                )
                
                # Compute metrics
                for metric in self._metrics:
                    metric_name = metric.__class__.__name__.replace("Metric", "").lower()
                    score = metric.compute(task, result)
                    result.metrics[metric_name] = score
                
                # Custom scorer
                if self._custom_scorer:
                    custom_score = self._custom_scorer(task, result)
                    result.metrics["custom"] = custom_score
                
                # Determine if passed (threshold-based)
                avg_score = sum(result.metrics.values()) / len(result.metrics) if result.metrics else 0.0
                result.passed = avg_score >= 0.7  # 70% threshold
                
            except Exception as e:
                result = EvaluationResult(
                    task_id=task.task_id,
                    actual_output="",
                    errors=[str(e)],
                    passed=False
                )
            
            results.append(result)
        
        return results
    
    def compute_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Compute summary statistics.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Summary dictionary
        """
        if not results:
            return {}
        
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        
        # Aggregate metrics
        metric_scores = {}
        for result in results:
            for metric_name, score in result.metrics.items():
                if metric_name not in metric_scores:
                    metric_scores[metric_name] = []
                metric_scores[metric_name].append(score)
        
        # Compute averages
        avg_metrics = {
            name: sum(scores) / len(scores)
            for name, scores in metric_scores.items()
        }
        
        # Average latency
        avg_latency = sum(r.latency for r in results) / total if results else 0.0
        
        # Total cost
        total_cost = sum(r.cost for r in results)
        
        return {
            "total_tasks": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "average_metrics": avg_metrics,
            "average_latency": avg_latency,
            "total_cost": total_cost
        }
