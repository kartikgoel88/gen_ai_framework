"""Planning Agent Example.

This example demonstrates how to use planning agents that create
step-by-step plans before execution.
"""

from src.framework.config import get_settings
from src.framework.api.deps import get_llm, get_rag
from src.framework.agents import (
    LangChainReActAgent,
    PlanningAgent,
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
        enable_web_search=True
    )
    
    # Create base agent
    base_agent = LangChainReActAgent(
        llm=llm,
        tools=tools
    )
    
    # Create planning agent
    planning_agent = PlanningAgent(
        base_agent=base_agent,
        enable_revision=True,
        max_revisions=2
    )
    
    # Complex task that requires planning
    task = "Research and write a comprehensive guide on implementing RAG systems, including best practices and common pitfalls"
    
    print(f"Task: {task}\n")
    print("=" * 60)
    print("Planning Agent Execution:")
    print("=" * 60 + "\n")
    
    # Execute with planning
    result = planning_agent.invoke(task)
    
    print("\n" + "=" * 60)
    print("Final Result:")
    print("=" * 60 + "\n")
    print(result[:500] + "..." if len(result) > 500 else result)


if __name__ == "__main__":
    main()
