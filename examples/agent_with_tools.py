"""Agent with Tools Example.

This example demonstrates how to:
1. Create an agent with RAG and web search tools
2. Use the agent to answer questions
3. Handle tool selection automatically
"""

from src.framework.config import get_settings
from src.framework.api.deps import get_agent, get_rag


def main():
    # Get settings
    settings = get_settings()
    
    # Get RAG client (must have documents ingested)
    rag = get_rag(settings)
    
    # Get agent (includes RAG + web search + MCP tools)
    agent = get_agent(settings, rag=rag, mcp=None)
    
    # Example questions
    questions = [
        "What documents have you ingested?",  # Uses RAG tool
        "What's the latest news about AI?",   # Uses web search tool
        "Find LinkedIn profile for John Doe", # Uses web search tool
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        try:
            answer = agent.invoke(question)
            print(f"Answer: {answer[:200]}...")
        except Exception as e:
            print(f"Error: {e}")
    
    # List available tools
    tools = agent.get_tools_description()
    print("\n\nAvailable tools:")
    for tool in tools:
        print(f"- {tool['name']}: {tool['description'][:80]}...")


if __name__ == "__main__":
    main()
