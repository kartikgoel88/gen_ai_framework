"""Graph Example - LangGraph Workflows.

This example demonstrates how to use LangGraph for building
stateful workflows with RAG and agents.

Key concepts:
- RAG Graph: Retrieve → Generate workflow
- Agent Graph: ReAct agent with tools
- State management: TypedDict for graph state
- Conditional flows: Dynamic routing based on state
"""

from src.framework.config import get_settings
from src.framework.api.deps import get_llm, get_rag
from src.framework.graph import build_rag_graph, build_agent_graph
from src.framework.adapters import LangChainLLMAdapter


def main():
    settings = get_settings()
    llm_client = get_llm(settings)
    rag = get_rag(settings)
    
    # Wrap LLM client as LangChain adapter for graph compatibility
    llm = LangChainLLMAdapter(llm_client=llm_client)
    
    print("=" * 60)
    print("Graph Example - LangGraph Workflows")
    print("=" * 60 + "\n")
    
    # Add some documents to RAG
    rag.add_documents(
        texts=[
            "Python is a high-level programming language known for its simplicity.",
            "Machine learning is a subset of artificial intelligence.",
            "LangGraph is a library for building stateful, multi-actor applications with LLMs."
        ],
        metadatas=[
            {"source": "python.txt"},
            {"source": "ai.txt"},
            {"source": "langgraph.txt"}
        ]
    )
    
    # 1. RAG Graph
    print("1. RAG Graph (Retrieve → Generate):")
    print("-" * 60)
    
    rag_graph = build_rag_graph(
        llm=llm,
        rag=rag,
        top_k=2,
        system_prompt="You are a helpful assistant. Use the context to answer questions."
    )
    
    # Invoke graph
    result = rag_graph.invoke({"query": "What is Python?"})
    print(f"Query: What is Python?")
    print(f"Context: {result.get('context', 'N/A')[:100]}...")
    print(f"Response: {result.get('response', 'N/A')}\n")
    
    # 2. Agent Graph (if available)
    print("2. Agent Graph (ReAct with Tools):")
    print("-" * 60)
    
    try:
        from src.framework.agents import build_framework_tools
        
        # Build tools for agent
        tools = build_framework_tools(
            rag_client=rag,
            mcp_client=None,
            enable_web_search=False  # Disable web search for this example
        )
        
        # Build agent graph
        agent_graph = build_agent_graph(
            llm=llm,
            tools=tools,
            system_prompt="You are a helpful assistant with access to document search."
        )
        
        # Invoke agent graph
        from langchain_core.messages import HumanMessage
        
        result = agent_graph.invoke({
            "messages": [HumanMessage(content="What documents have you ingested?")]
        })
        
        # Extract final message
        messages = result.get("messages", [])
        if messages:
            final_message = messages[-1]
            if hasattr(final_message, "content"):
                print(f"Agent Response: {final_message.content}\n")
            else:
                print(f"Agent Response: {final_message}\n")
        else:
            print("No response from agent\n")
            
    except Exception as e:
        print(f"Agent graph not available: {e}\n")
    
    # 3. Streaming Graph Execution
    print("3. Streaming Graph Execution:")
    print("-" * 60)
    
    print("Streaming RAG graph:")
    for chunk in rag_graph.stream({"query": "What is machine learning?"}):
        # Process each node's output
        for node_name, node_output in chunk.items():
            if node_name == "generate" and "response" in node_output:
                print(f"Generated: {node_output['response']}")
            elif node_name == "retrieve" and "context" in node_output:
                print(f"Retrieved context (length: {len(node_output.get('context', ''))})")
    print()
    
    # 4. Graph State Inspection
    print("4. Graph State Inspection:")
    print("-" * 60)
    
    result = rag_graph.invoke({"query": "Tell me about LangGraph"})
    
    print("Graph State:")
    print(f"  Query: {result.get('query', 'N/A')}")
    print(f"  Context length: {len(result.get('context', ''))}")
    print(f"  Response length: {len(result.get('response', ''))}")
    print(f"  Response preview: {result.get('response', 'N/A')[:100]}...\n")
    
    # 5. Multiple Queries with State
    print("5. Multiple Queries (State Persistence):")
    print("-" * 60)
    
    queries = [
        "What is Python?",
        "How does it relate to machine learning?",
        "What about LangGraph?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"Query {i}: {query}")
        result = rag_graph.invoke({"query": query})
        print(f"  Response: {result.get('response', 'N/A')[:80]}...\n")


if __name__ == "__main__":
    main()
