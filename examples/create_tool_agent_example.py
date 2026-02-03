"""Create Tool Agent Example.

Demonstrates the create_tool_agent factory function for easily creating
agents with RAG, web search, and MCP tools.
"""

from src.framework.config import get_settings
from src.framework.api.deps import get_llm, get_rag
from src.framework.agents import create_tool_agent


def main():
    # Get settings and components
    settings = get_settings()
    llm = get_llm(settings)
    rag = get_rag(settings)
    
    # Create agent with tools using the factory function
    # This automatically:
    # - Wraps LLMClient as LangChain model
    # - Builds RAG, web search, and MCP tools
    # - Configures the agent
    agent = create_tool_agent(
        llm=llm,
        rag_client=rag,
        enable_web_search=True,
        system_prompt="You are a helpful research assistant with access to documents and web search.",
        verbose=settings.DEBUG,
    )
    
    print("=" * 60)
    print("Tool Agent Example")
    print("=" * 60 + "\n")
    
    # Example questions that will use different tools
    questions = [
        "What documents have you ingested?",  # Uses RAG tool
        "What's the latest news about AI?",   # Uses web search tool
        "Find LinkedIn profile for John Doe", # Uses web search tool
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 60)
        try:
            answer = agent.invoke(question)
            print(f"Answer: {answer[:300]}...")
        except Exception as e:
            print(f"Error: {e}")
    
    # List available tools
    print("\n" + "=" * 60)
    print("Available Tools:")
    print("=" * 60)
    tools = agent.get_tools_description()
    for tool in tools:
        print(f"- {tool['name']}: {tool['description'][:80]}...")


if __name__ == "__main__":
    main()
