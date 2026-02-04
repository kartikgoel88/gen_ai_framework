"""Agent Flow Example - Understanding the Complete Agent Flow.

This example demonstrates the complete agent flow from user message
to final response, showing each step in the process.

Flow:
1. User message → PromptBuilder
2. Messages → LangChain Agent Graph
3. LLM Adapter → LLM Provider
4. Tool Calls → Tool Interpreter
5. Tool Execution → Results
6. Final Response → User
"""

from src.framework.config import get_settings
from src.framework.api.deps import get_llm, get_rag
from src.framework.agents import create_tool_agent
from src.framework.adapters import LangChainLLMAdapter


def main():
    settings = get_settings()
    llm = get_llm(settings)
    rag = get_rag(settings)
    
    print("=" * 60)
    print("Agent Flow Example - Complete Flow Demonstration")
    print("=" * 60 + "\n")
    
    # Add documents to RAG
    rag.add_documents(
        texts=[
            "Python is a programming language created by Guido van Rossum.",
            "The framework supports multiple LLM providers: OpenAI, Gemini, Grok.",
            "Agents can use tools like RAG search, web search, and MCP tools."
        ],
        metadatas=[
            {"source": "python.txt"},
            {"source": "llm.txt"},
            {"source": "agents.txt"}
        ]
    )
    
    # Create agent with verbose logging
    agent = create_tool_agent(
        llm=llm,
        rag_client=rag,
        enable_web_search=False,  # Disable for cleaner output
        system_prompt="You are a helpful assistant. Always use tools when needed.",
        verbose=True  # Enable verbose logging
    )
    
    print("=" * 60)
    print("Step-by-Step Agent Flow")
    print("=" * 60 + "\n")
    
    # Example 1: Simple question (no tools needed)
    print("Example 1: Simple Question (No Tools)")
    print("-" * 60)
    print("User: 'Hello, how are you?'")
    print("\nFlow:")
    print("  1. PromptBuilder creates messages")
    print("  2. Agent graph receives messages")
    print("  3. LLM generates response (no tool calls)")
    print("  4. Response returned to user")
    print("\nResponse:")
    try:
        response = agent.invoke("Hello, how are you?")
        print(f"  {response}\n")
    except Exception as e:
        print(f"  Error: {e}\n")
    
    # Example 2: Question requiring RAG tool
    print("Example 2: Question Requiring RAG Tool")
    print("-" * 60)
    print("User: 'What is Python?'")
    print("\nFlow:")
    print("  1. PromptBuilder creates messages with tool descriptions")
    print("  2. Agent graph receives messages")
    print("  3. LLM decides to use rag_search tool")
    print("  4. Tool call detected: {'name': 'rag_search', 'arguments': {...}}")
    print("  5. ToolInterpreter validates and executes tool")
    print("  6. RAG search returns document chunks")
    print("  7. Tool results added to conversation")
    print("  8. LLM generates final answer using tool results")
    print("  9. Response returned to user")
    print("\nResponse:")
    try:
        response = agent.invoke("What is Python?")
        print(f"  {response[:200]}...\n")
    except Exception as e:
        print(f"  Error: {e}\n")
    
    # Example 3: Multi-turn conversation
    print("Example 3: Multi-Turn Conversation")
    print("-" * 60)
    
    conversation = [
        "What documents have you ingested?",
        "Tell me more about Python",
        "What about the framework?"
    ]
    
    chat_history = []
    
    for i, message in enumerate(conversation, 1):
        print(f"\nTurn {i}:")
        print(f"  User: {message}")
        
        try:
            response = agent.invoke(
                message,
                chat_history=chat_history
            )
            print(f"  Agent: {response[:150]}...")
            
            # Add to history
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("Agent Components")
    print("=" * 60 + "\n")
    
    # Show available tools
    tools = agent.get_tools_description()
    print("Available Tools:")
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description'][:60]}...")
    
    print("\n" + "=" * 60)
    print("Understanding the Flow")
    print("=" * 60 + "\n")
    
    print("""
The agent flow consists of these key components:

1. PromptBuilder
   - Combines system prompt, chat history, and user message
   - Adds tool descriptions to system prompt
   - Creates LangChain message objects

2. LangChain Agent Graph
   - Manages the ReAct loop (Reasoning + Acting)
   - Decides when to use tools
   - Coordinates LLM calls and tool execution

3. LangChainLLMAdapter
   - Wraps framework LLMClient as LangChain BaseChatModel
   - Converts LangChain messages to provider format
   - Handles native tool calling if available

4. ToolInterpreter
   - Parses tool calls from LLM responses
   - Validates tool calls against registry
   - Executes tools and returns results

5. Tool Execution
   - RAG search: Retrieves relevant documents
   - Web search: Searches the internet
   - MCP tools: Calls external tools via MCP protocol

6. Response Generation
   - LLM receives tool results
   - Generates final answer
   - Returns to user

For more details, see README.md "Agent Architecture & Flow" section.
""")


if __name__ == "__main__":
    main()
