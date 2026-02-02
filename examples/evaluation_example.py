"""Agent Evaluation Example.

This example demonstrates how to evaluate agent performance
using the evaluation framework.
"""

from src.framework.config import get_settings
from src.framework.api.deps import get_llm, get_rag
from src.framework.agents import (
    LangChainReActAgent,
    AgentEvaluator,
    EvaluationTask,
    build_framework_tools
)


def main():
    # Get settings and components
    settings = get_settings()
    llm = get_llm(settings)
    rag = get_rag(settings)
    
    # Build tools
    tools = build_framework_tools(
        rag_client=rag,
        mcp_client=None,
        enable_web_search=False
    )
    
    # Create agent
    agent = LangChainReActAgent(llm=llm, tools=tools)
    
    # Create evaluator
    evaluator = AgentEvaluator()
    
    # Create evaluation tasks
    tasks = [
        EvaluationTask(
            task_id="task1",
            prompt="What is Python?",
            expected_output="Python is a programming language",
            expected_tools=["rag_search"]
        ),
        EvaluationTask(
            task_id="task2",
            prompt="Explain machine learning",
            expected_output="Machine learning is a subset of AI",
            expected_tools=["rag_search"]
        ),
    ]
    
    print("=" * 60)
    print("Agent Evaluation")
    print("=" * 60 + "\n")
    
    # Run evaluation
    results = evaluator.evaluate(agent, tasks)
    
    # Display results
    for result in results:
        print(f"Task: {result.task_id}")
        print(f"  Output: {result.actual_output[:100]}...")
        print(f"  Metrics: {result.metrics}")
        print(f"  Tool Usage: {result.tool_usage}")
        print(f"  Latency: {result.latency:.2f}s")
        print(f"  Passed: {result.passed}\n")
    
    # Summary
    summary = evaluator.compute_summary(results)
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Total Tasks: {summary['total_tasks']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Pass Rate: {summary['pass_rate']:.2%}")
    print(f"Average Metrics: {summary['average_metrics']}")
    print(f"Average Latency: {summary['average_latency']:.2f}s")


if __name__ == "__main__":
    main()
