"""Complete Agent Features Example.

This example demonstrates combining multiple agent features:
- Memory
- Monitoring
- Cost Tracking
- Personas
- Error Recovery
"""

from src.framework.config import get_settings
from src.framework.api.deps import get_llm, get_rag
from src.framework.agents import (
    LangChainReActAgent,
    AgentWithMemory,
    RAGMemoryStore,
    MonitoredAgent,
    AgentMonitor,
    CostTrackingAgent,
    CostTracker,
    PersonaAgent,
    PersonaType,
    ErrorRecoveryAgent,
    RetryConfig,
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
    
    # Add memory
    memory = RAGMemoryStore(rag_client=rag)
    agent_with_memory = AgentWithMemory(
        base_agent=base_agent,
        memory=memory
    )
    
    # Add monitoring
    monitor = AgentMonitor()
    monitored_agent = MonitoredAgent(
        base_agent=agent_with_memory,
        monitor=monitor,
        agent_id="demo_agent"
    )
    
    # Add cost tracking
    cost_tracker = CostTracker()
    cost_tracker.set_budget("demo_agent", budget=1.0)  # $1 budget
    
    cost_tracking_agent = CostTrackingAgent(
        base_agent=monitored_agent,
        cost_tracker=cost_tracker,
        agent_id="demo_agent",
        model=settings.LLM_MODEL
    )
    
    # Add persona
    persona_agent = PersonaAgent(
        base_agent=cost_tracking_agent,
        persona=PersonaType.RESEARCHER.value  # Use researcher persona
    )
    
    # Add error recovery
    retry_config = RetryConfig(
        max_retries=2,
        backoff_factor=1.5
    )
    
    final_agent = ErrorRecoveryAgent(
        base_agent=persona_agent,
        retry_config=retry_config
    )
    
    # Use the fully-featured agent
    question = "What are the key components of a RAG system?"
    user_id = "demo_user"
    
    print("=" * 60)
    print("Complete Agent Features Demo")
    print("=" * 60 + "\n")
    print(f"Question: {question}\n")
    
    response = final_agent.invoke(question)
    
    print(f"Response: {response[:300]}...\n")
    
    # Show metrics
    print("=" * 60)
    print("Metrics:")
    print("=" * 60)
    metrics = monitor.get_metrics("demo_agent")
    if metrics:
        print(f"Total Invocations: {metrics.total_invocations}")
        print(f"Total Tool Calls: {metrics.total_tool_calls}")
        print(f"Tool Usage: {dict(metrics.tool_usage)}")
        print(f"Average Latency: {metrics.average_latency:.2f}s")
    
    # Show cost
    print("\n" + "=" * 60)
    print("Cost Summary:")
    print("=" * 60)
    cost_summary = cost_tracker.get_cost_summary("demo_agent")
    print(f"Total Cost: ${cost_summary['total_cost']:.4f}")
    print(f"Total Operations: {cost_summary['total_operations']}")
    print(f"Total Tokens: {cost_summary['total_input_tokens'] + cost_summary['total_output_tokens']}")


if __name__ == "__main__":
    main()
