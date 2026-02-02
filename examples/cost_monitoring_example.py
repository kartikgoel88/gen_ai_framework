"""Cost Tracking and Monitoring Example.

This example demonstrates cost tracking and monitoring features.
"""

from src.framework.config import get_settings
from src.framework.api.deps import get_llm, get_rag
from src.framework.agents import (
    LangChainReActAgent,
    CostTrackingAgent,
    CostTracker,
    MonitoredAgent,
    AgentMonitor,
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
    
    # Create base agent
    base_agent = LangChainReActAgent(llm=llm, tools=tools)
    
    # Add monitoring
    monitor = AgentMonitor()
    monitored_agent = MonitoredAgent(
        base_agent=base_agent,
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
    
    print("=" * 60)
    print("Cost Tracking and Monitoring")
    print("=" * 60 + "\n")
    
    # Run some queries
    questions = [
        "What is Python?",
        "Explain machine learning",
        "What is RAG?",
    ]
    
    for question in questions:
        print(f"Question: {question}")
        try:
            response = cost_tracking_agent.invoke(question)
            print(f"Response: {response[:100]}...\n")
        except Exception as e:
            print(f"Error: {e}\n")
    
    # Show metrics
    print("=" * 60)
    print("Monitoring Metrics:")
    print("=" * 60)
    metrics = monitor.get_metrics("demo_agent")
    if metrics:
        print(f"Total Invocations: {metrics.total_invocations}")
        print(f"Total Tool Calls: {metrics.total_tool_calls}")
        print(f"Tool Usage: {dict(metrics.tool_usage)}")
        print(f"Average Latency: {metrics.average_latency:.2f}s")
        print(f"Error Rate: {metrics.error_rate:.2%}\n")
    
    # Show cost
    print("=" * 60)
    print("Cost Summary:")
    print("=" * 60)
    cost_summary = cost_tracker.get_cost_summary("demo_agent")
    print(f"Total Cost: ${cost_summary['total_cost']:.4f}")
    print(f"Total Operations: {cost_summary['total_operations']}")
    print(f"Total Input Tokens: {cost_summary['total_input_tokens']}")
    print(f"Total Output Tokens: {cost_summary['total_output_tokens']}")
    print(f"By Model: {cost_summary['by_model']}")


if __name__ == "__main__":
    main()
