"""Streaming Agent Example.

This example demonstrates how to use the streaming agent to see
intermediate steps, tool calls, and reasoning in real-time.
"""

from src.framework.config import get_settings
from src.framework.api.deps import get_llm, get_rag
from src.framework.agents.streaming_agent import StreamingAgent, AgentEventType
from src.framework.agents.tools import build_framework_tools


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
    
    # Create streaming agent
    agent = StreamingAgent(
        llm=llm,
        tools=tools,
        system_prompt="You are a helpful AI assistant. Show your reasoning process."
    )
    
    # Example question
    question = "What is RAG and how does it work?"
    
    print(f"Question: {question}\n")
    print("=" * 60)
    print("Agent Execution Stream:")
    print("=" * 60 + "\n")
    
    # Stream agent execution
    for event in agent.invoke_stream(question):
        event_type = event.type.value.upper()
        content = event.content
        
        # Format output based on event type
        if event.type == AgentEventType.THINKING:
            print(f"🤔 [{event_type}] {content}")
        elif event.type == AgentEventType.TOOL_SELECTION:
            print(f"🔧 [{event_type}] {content}")
        elif event.type == AgentEventType.TOOL_CALL:
            print(f"📞 [{event_type}] {content}")
        elif event.type == AgentEventType.TOOL_RESULT:
            print(f"📋 [{event_type}] {content[:100]}...")
        elif event.type == AgentEventType.REASONING:
            print(f"💭 [{event_type}] {content}")
        elif event.type == AgentEventType.RESPONSE_CHUNK:
            print(content, end="", flush=True)
        elif event.type == AgentEventType.COMPLETE:
            print(f"\n\n✅ [{event_type}] {content}")
        elif event.type == AgentEventType.ERROR:
            print(f"\n❌ [{event_type}] {content}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
