"""Error Recovery Example.

This example demonstrates error recovery and retry logic.
"""

from src.framework.config import get_settings
from src.framework.api.deps import get_llm, get_rag
from src.framework.agents import (
    LangChainReActAgent,
    ErrorRecoveryAgent,
    RetryConfig,
    ErrorType,
    create_simple_fallback,
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
    
    # Configure retry
    retry_config = RetryConfig(
        max_retries=3,
        backoff_factor=2.0,
        initial_delay=1.0,
        retryable_errors=[ErrorType.NETWORK, ErrorType.RATE_LIMIT, ErrorType.TIMEOUT]
    )
    
    # Create error recovery agent
    recovery_agent = ErrorRecoveryAgent(
        base_agent=base_agent,
        retry_config=retry_config,
        fallback_strategy=create_simple_fallback
    )
    
    print("=" * 60)
    print("Error Recovery Example")
    print("=" * 60 + "\n")
    
    print("Retry Configuration:")
    print(f"  Max Retries: {retry_config.max_retries}")
    print(f"  Backoff Factor: {retry_config.backoff_factor}")
    print(f"  Initial Delay: {retry_config.initial_delay}s")
    print(f"  Retryable Errors: {[e.value for e in retry_config.retryable_errors]}\n")
    
    print("Error Recovery Agent:")
    print("  - Automatically retries on retryable errors")
    print("  - Uses exponential backoff")
    print("  - Falls back to error message if all retries fail")
    print("  - Classifies errors (network, rate limit, timeout, etc.)\n")
    
    # Example: Normal invocation (would work if API keys are set)
    print("Example Usage:")
    print("  agent = ErrorRecoveryAgent(base_agent, retry_config)")
    print("  response = agent.invoke('Hello!')")
    print("  # Automatically retries on network errors, rate limits, etc.")


if __name__ == "__main__":
    main()
