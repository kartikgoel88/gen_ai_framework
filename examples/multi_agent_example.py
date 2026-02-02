"""Multi-Agent System Example.

This example demonstrates how to use multiple specialized agents
working together to accomplish complex tasks.
"""

from src.framework.config import get_settings
from src.framework.api.deps import get_llm, get_rag
from src.framework.agents import (
    LangChainReActAgent,
    MultiAgentSystem,
    ResearcherAgent,
    WriterAgent,
    ReviewerAgent,
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
    
    # Create specialized agents
    researcher = ResearcherAgent(base_agent)
    writer = WriterAgent(base_agent)
    reviewer = ReviewerAgent(base_agent)
    
    # Create multi-agent system
    system = MultiAgentSystem(
        agents={
            "researcher": researcher,
            "writer": writer,
            "reviewer": reviewer
        }
    )
    
    # Example task
    task = "Write a comprehensive guide on RAG (Retrieval-Augmented Generation) systems"
    
    print(f"Task: {task}\n")
    print("=" * 60)
    print("Multi-Agent Execution:")
    print("=" * 60 + "\n")
    
    # Execute multi-agent system
    result = system.invoke(task)
    
    print("\n" + "=" * 60)
    print("Final Result:")
    print("=" * 60 + "\n")
    print(result)


if __name__ == "__main__":
    main()
