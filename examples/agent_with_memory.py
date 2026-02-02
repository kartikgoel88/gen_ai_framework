"""Agent with Memory Example.

This example demonstrates how to use agents with persistent memory
to remember past conversations.
"""

from src.framework.config import get_settings
from src.framework.api.deps import get_llm, get_rag
from src.framework.agents import (
    LangChainReActAgent,
    AgentWithMemory,
    RAGMemoryStore,
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
        enable_web_search=False  # Just use RAG for this example
    )
    
    # Create base agent
    base_agent = LangChainReActAgent(
        llm=llm,
        tools=tools
    )
    
    # Create memory store
    memory = RAGMemoryStore(rag_client=rag)
    
    # Create agent with memory
    agent = AgentWithMemory(
        base_agent=base_agent,
        memory=memory
    )
    
    user_id = "user123"
    
    # First conversation
    print("=" * 60)
    print("Conversation 1:")
    print("=" * 60)
    question1 = "What is my favorite programming language?"
    print(f"User: {question1}")
    
    # Store initial preference
    memory.store(
        user_id=user_id,
        message="My favorite programming language is Python",
        response="Got it! I'll remember that Python is your favorite programming language.",
        metadata={"type": "preference"}
    )
    
    response1 = agent.invoke_with_memory(question1, user_id=user_id)
    print(f"Agent: {response1}\n")
    
    # Second conversation (should remember)
    print("=" * 60)
    print("Conversation 2 (later):")
    print("=" * 60)
    question2 = "What did I tell you about my favorite language?"
    print(f"User: {question2}")
    
    response2 = agent.invoke_with_memory(question2, user_id=user_id)
    print(f"Agent: {response2}\n")
    
    # Show retrieved memories
    print("=" * 60)
    print("Retrieved Memories:")
    print("=" * 60)
    memories = memory.retrieve(user_id, question2, top_k=3)
    for i, mem in enumerate(memories, 1):
        print(f"\nMemory {i}:")
        print(f"  User: {mem.message}")
        print(f"  Agent: {mem.response}")
        print(f"  Time: {mem.timestamp}")


if __name__ == "__main__":
    main()
