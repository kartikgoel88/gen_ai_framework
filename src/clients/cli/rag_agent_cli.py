"""Unified CLI for RAG and Agent operations.

Combines RAG operations (ingest, query, clear, export) with agent operations
(invoke, tools) in a single command-line interface. Includes an interactive mode
for conversational interaction.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from src.framework.config import get_settings
from src.framework.api.deps import get_llm, get_rag, get_mcp_client
from src.framework.agents import create_tool_agent
from src.framework.utils.debug import set_debug_enabled, get_debug_logger


# RAG Commands

def ingest_command(args):
    """Ingest documents into RAG."""
    settings = get_settings()
    rag = get_rag(settings)
    
    if args.file:
        # Read from file
        text = Path(args.file).read_text()
        metadata = {"source": str(args.file)}
    elif args.text:
        text = args.text
        metadata = {}
    else:
        print("Error: Provide either --file or --text", file=sys.stderr)
        return 1
    
    if args.metadata:
        try:
            metadata.update(json.loads(args.metadata))
        except json.JSONDecodeError:
            print("Warning: Invalid JSON metadata, ignoring", file=sys.stderr)
    
    rag.add_documents([text], metadatas=[metadata])
    print(f"✅ Ingested document into RAG")
    return 0


def query_command(args):
    """Query RAG."""
    settings = get_settings()
    rag = get_rag(settings)
    
    if args.llm:
        llm = get_llm(settings)
        answer = rag.query(args.question, llm_client=llm, top_k=args.top_k)
        print(f"Answer: {answer}")
    else:
        chunks = rag.retrieve(args.question, top_k=args.top_k)
        print(f"Retrieved {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n{i}. {chunk.get('content', '')[:200]}...")
            if chunk.get('metadata'):
                print(f"   Metadata: {chunk['metadata']}")
    
    return 0


def clear_command(args):
    """Clear RAG store."""
    settings = get_settings()
    rag = get_rag(settings)
    
    rag.clear()
    print("✅ Cleared RAG store")
    return 0


def export_command(args):
    """Export RAG corpus."""
    settings = get_settings()
    rag = get_rag(settings)
    
    corpus = rag.export_corpus(format=args.format)
    
    if args.output:
        output_path = Path(args.output)
        if args.format == "jsonl":
            with output_path.open("w") as f:
                for item in corpus:
                    f.write(json.dumps(item) + "\n")
        else:
            output_path.write_text(json.dumps(corpus, indent=2))
        print(f"✅ Exported to {args.output}")
    else:
        print(json.dumps(corpus, indent=2))
    
    return 0


# Agent Commands

def agent_invoke_command(args):
    """Invoke agent with RAG and tools."""
    settings = get_settings()
    llm = get_llm(settings)
    rag = get_rag(settings)
    mcp = None
    if args.mcp:
        try:
            mcp = get_mcp_client(settings)
        except Exception as e:
            print(f"Warning: Could not initialize MCP client: {e}", file=sys.stderr)
            print("Continuing without MCP tools...", file=sys.stderr)
    
    # Create agent with tools
    agent = create_tool_agent(
        llm=llm,
        rag_client=rag,
        mcp_client=mcp,
        enable_web_search=args.web_search,
        system_prompt=args.system_prompt,
        verbose=args.verbose,
    )
    
    # Get message
    if args.file:
        message = Path(args.file).read_text()
    else:
        message = args.message
    
    # Invoke agent
    response = agent.invoke(message, system_prompt=args.system_prompt)
    
    # Output result
    if args.output:
        Path(args.output).write_text(response)
        print(f"✅ Response written to {args.output}")
    else:
        print(response)
    
    return 0


def agent_tools_command(args):
    """List agent tools."""
    settings = get_settings()
    llm = get_llm(settings)
    rag = get_rag(settings)
    mcp = None
    if args.mcp:
        try:
            mcp = get_mcp_client(settings)
        except Exception as e:
            print(f"Warning: Could not initialize MCP client: {e}", file=sys.stderr)
            print("Continuing without MCP tools...", file=sys.stderr)
    
    # Create agent to get tools
    agent = create_tool_agent(
        llm=llm,
        rag_client=rag,
        mcp_client=mcp,
        enable_web_search=args.web_search,
    )
    
    tools = agent.get_tools_description()
    
    if args.json:
        print(json.dumps(tools, indent=2))
    else:
        print("Available Tools:")
        for tool in tools:
            print(f"\n  {tool['name']}")
            print(f"    {tool['description']}")
    
    return 0


def interactive_command(args):
    """Start interactive mode for conversational interaction."""
    settings = get_settings()
    
    # Enable debug if verbose flag is set (or if DEBUG is enabled in config)
    debug_logger = get_debug_logger(enabled=args.verbose if args.verbose else None)
    if args.verbose:
        set_debug_enabled(True)
    
    llm = get_llm(settings)
    rag = get_rag(settings)
    
    # Initialize MCP if requested
    mcp = None
    if args.mcp:
        try:
            mcp = get_mcp_client(settings)
        except Exception as e:
            print(f"Warning: Could not initialize MCP client: {e}", file=sys.stderr)
            print("Continuing without MCP tools...", file=sys.stderr)
    
    # Define helper function to check RAG status
    def check_rag_status():
        """Check if RAG has documents."""
        try:
            # Try a simple query to see if there's data
            test_chunks = rag.retrieve("test", top_k=1)
            return len(test_chunks) > 0
        except:
            return False
    
    # Create agent with improved default system prompt if none provided
    default_system_prompt = args.system_prompt
    if not default_system_prompt:
        has_docs = check_rag_status()
        default_system_prompt = (
            "You are a helpful AI assistant with access to tools. "
            "IMPORTANT: You MUST use tools to answer questions. Do not guess or make up answers.\n\n"
            "Available tools:\n"
            "- rag_search: Search your knowledge base for information from ingested documents. "
            "  USE THIS FIRST when questions relate to documents, information, or topics that might be in your knowledge base.\n"
            f"{'- web_search: Find current information from the internet.\n' if args.web_search else ''}\n"
            "Workflow:\n"
            "1. If the question relates to documents or your knowledge base, ALWAYS call rag_search first.\n"
            "2. Use the results from rag_search to answer the question.\n"
            f"{'3. Only use web_search if you need current information not in your documents.\n' if args.web_search else ''}\n"
            f"{'Your knowledge base has documents available - use rag_search to access them. ' if has_docs else 'Note: Your knowledge base may be empty - still try rag_search first, then use web_search if needed. '}\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- When user asks about ANY information, you MUST call rag_search first.\n"
            "- Extract keywords from the question and use them as the query.\n"
            "- Examples: 'Kartik qualifications' → rag_search(query='Kartik qualifications')\n"
            "- Examples: 'Kartik number' → rag_search(query='Kartik number')\n"
            "- DO NOT refuse to use tools. DO NOT say you can't assist.\n"
            "- ALWAYS call rag_search when asked about information, even if you think it might not be there."
        )
    
    # Create agent
    try:
        agent = create_tool_agent(
            llm=llm,
            rag_client=rag,
            mcp_client=mcp,
            enable_web_search=args.web_search,
            system_prompt=default_system_prompt,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"❌ Error initializing agent: {type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    # Mode: 'agent' or 'rag'
    mode = args.initial_mode or 'agent'
    chat_history = []
    
    # Runtime settings (can be toggled interactively)
    # Use list to avoid scope issues when modifying
    runtime_settings = {
        'rag_llm': args.rag_llm,  # Can be toggled with /toggle-llm
    }
    
    def rag_workflow_welcome():
        """Show RAG workflow welcome and guide."""
        has_docs = check_rag_status()
        
        print("\n" + "=" * 70)
        print("📚 RAG WORKFLOW MODE")
        print("=" * 70)
        
        # Show enabled features/flags
        print("\n📋 Configuration:")
        print("   ✅ RAG (Document Search)  - Enabled")
        if runtime_settings['rag_llm']:
            print("   ✅ LLM Answers            - Enabled (will generate answers)")
            print("      💡 Use /toggle-llm to disable and show chunks instead")
        else:
            print("   ⚙️  LLM Answers            - Disabled (shows chunks only)")
            print("      💡 Use /toggle-llm to enable LLM-generated answers")
        print(f"   ⚙️  Top-K Retrieval        - {args.rag_top_k} chunks")
        if debug_logger._get_enabled():
            print("   ✅ Debug Mode              - Enabled")
        
        if not has_docs:
            print("\n⚠️  No documents found in your knowledge base.")
            print("\n📝 Step 1: Ingest Documents")
            print("   You can ingest documents in two ways:")
            print("   1. Exit and use CLI: rag-agent-cli rag ingest --file document.txt")
            print("   2. Or provide a file path now (type: /ingest <file_path>)")
            print("\n   Example: /ingest ./documents/readme.txt")
        else:
            print("\n✅ Knowledge base ready! You have documents ingested.")
            print("\n🔍 Step 2: Query Your Documents")
            print("   Simply ask questions about your documents.")
            print("   Examples:")
            print("     • 'What is machine learning?'")
            print("     • 'Explain RAG systems'")
            print("     • 'What documents mention Python?'")
        
        print("\n💡 Quick Commands:")
        print("   /ingest <file>  - Ingest a document file")
        print("   /toggle-llm     - Toggle LLM answers on/off")
        print("   /status         - Check knowledge base status")
        print("   /config         - Show current configuration/flags")
        print("   /help           - Show this help")
        print("   /agent          - Switch to chat mode")
        print("   /exit           - Exit")
        print("\n💡 Tip: Use /toggle-llm to switch between showing chunks or LLM-generated answers")
        print("=" * 70 + "\n")
    
    def agent_chat_welcome():
        """Show agent chat welcome."""
        has_docs = check_rag_status()
        
        print("\n" + "=" * 70)
        print("💬 CHAT MODE - Conversational AI Assistant")
        print("=" * 70)
        
        # Show enabled features/flags
        print("\n📋 Enabled Features:")
        print("   ✅ RAG (Document Search)  - Search your ingested documents")
        if has_docs:
            print("      📚 Knowledge base has documents available")
        else:
            print("      ⚠️  No documents in knowledge base yet")
        if args.web_search:
            print("   ✅ Web Search              - Search the internet")
        else:
            print("   ❌ Web Search              - Disabled")
        if mcp:
            print("   ✅ MCP Tools               - Available")
        else:
            print("   ❌ MCP Tools               - Not configured")
        if args.memory:
            print("   ✅ Conversation Memory     - Enabled")
        else:
            print("   ❌ Conversation Memory     - Disabled")
        if debug_logger._get_enabled():
            print("   ✅ Debug Mode              - Enabled")
        
        print("\n👋 Hi! I'm your AI assistant. I can help you with:")
        print("   • Answering questions using your documents (via RAG search)")
        if args.web_search:
            print("   • Searching the web for current information")
        if mcp:
            print("   • Using MCP tools to help you accomplish tasks")
        
        print("\n💬 Just start chatting! The agent will automatically use RAG when needed.")
        print("   Try asking:")
        if has_docs:
            print("   • 'What information do you have about [topic]?'")
            print("   • 'Search your documents for [query]'")
            print("   • 'What do your documents say about [subject]?'")
        else:
            print("   • 'What documents have you ingested?' (will show no docs)")
            print("   💡 Tip: Ingest documents first using /rag mode or CLI")
        if args.web_search:
            print("   • 'What's the latest news about AI?'")
        print("   • 'Help me understand machine learning'")
        
        print("\n💡 Commands:")
        print("   /help     - Show help")
        print("   /tools    - List available tools (including rag_search)")
        print("   /rag-status - Check RAG knowledge base status")
        print("   /test-rag <query> - Test RAG tool directly (bypasses agent)")
        print("   /config   - Show current configuration/flags")
        print("   /rag      - Switch to RAG workflow mode")
        print("   /clear    - Clear conversation history")
        print("   /exit     - Exit")
        print("\n💡 Tips for using RAG:")
        print("   • Be specific: 'Search your documents for information about X'")
        print("   • Ask directly: 'What do your documents say about Y?'")
        print("   • Use /test-rag to verify RAG is working")
        print("=" * 70 + "\n")
    
    def show_mode_help(current_mode):
        """Show mode-specific help."""
        if current_mode == 'agent':
            agent_chat_welcome()
        else:
            rag_workflow_welcome()
    
    # Show initial welcome based on mode
    if mode == 'agent':
        agent_chat_welcome()
    else:
        rag_workflow_welcome()
    
    try:
        while True:
            try:
                # Get user input with mode-specific prompts
                if mode == 'agent':
                    prompt = "💬 You: "
                else:
                    prompt = "🔍 Query: "
                user_input = input(prompt).strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    cmd = user_input[1:].lower().split()[0] if user_input[1:] else ''
                    
                    if cmd == 'exit' or cmd == 'quit':
                        print("\n👋 Goodbye!")
                        break
                    
                    elif cmd == 'help':
                        show_mode_help(mode)
                    
                    elif cmd == 'mode':
                        mode = 'rag' if mode == 'agent' else 'agent'
                        print(f"\n🔄 Switching to {mode.upper()} mode...")
                        show_mode_help(mode)
                    
                    elif cmd == 'toggle-llm' and mode == 'rag':
                        # Toggle LLM answers on/off
                        runtime_settings['rag_llm'] = not runtime_settings['rag_llm']
                        if runtime_settings['rag_llm']:
                            print("\n✅ LLM answers enabled - queries will generate full answers")
                            print("   Use /toggle-llm again to switch back to showing chunks only\n")
                        else:
                            print("\n✅ LLM answers disabled - queries will show document chunks")
                            print("   Use /toggle-llm again to enable LLM-generated answers\n")
                        continue
                    
                    elif cmd == 'status' and mode == 'rag':
                        # Show RAG status
                        has_docs = check_rag_status()
                        if has_docs:
                            try:
                                chunks = rag.retrieve("", top_k=100)  # Get a sample
                                print(f"\n✅ Knowledge Base Status:")
                                print(f"   Documents available: Yes")
                                print(f"   Sample chunks found: {len(chunks)}")
                                print(f"   LLM Answers: {'✅ Enabled' if runtime_settings['rag_llm'] else '❌ Disabled (chunks only)'}")
                                print("   You can query your documents!\n")
                            except:
                                print("\n✅ Knowledge base is ready.\n")
                        else:
                            print("\n⚠️  No documents in knowledge base.")
                            print("   Use /ingest <file_path> to add documents.\n")
                        continue
                    
                    elif cmd == 'rag':
                        mode = 'rag'
                        print(f"\n🔄 Switching to RAG workflow mode...")
                        show_mode_help(mode)
                    
                    elif cmd == 'agent':
                        mode = 'agent'
                        print(f"\n🔄 Switching to chat mode...")
                        show_mode_help(mode)
                    
                    elif cmd == 'ingest' and mode == 'rag':
                        # Handle inline ingest
                        parts = user_input.split(None, 1)
                        if len(parts) < 2:
                            print("\n❌ Please provide a file path: /ingest <file_path>")
                            print("   Example: /ingest ./documents/readme.txt\n")
                        else:
                            file_path = Path(parts[1].strip().strip('"\''))
                            if file_path.exists():
                                try:
                                    text = file_path.read_text()
                                    metadata = {"source": str(file_path)}
                                    rag.add_documents([text], metadatas=[metadata])
                                    print(f"\n✅ Successfully ingested: {file_path}")
                                    print("   You can now query this document!\n")
                                except Exception as e:
                                    print(f"\n❌ Error ingesting file: {e}\n")
                            else:
                                print(f"\n❌ File not found: {file_path}\n")
                        continue
                    
                    elif cmd == 'tools':
                        tools = agent.get_tools_description()
                        print("\n" + "=" * 70)
                        print("🔧 Available Tools")
                        print("=" * 70)
                        for tool in tools:
                            print(f"\n  • {tool['name']}")
                            print(f"    {tool['description']}")
                            if tool['name'] == 'rag_search':
                                has_docs = check_rag_status()
                                if has_docs:
                                    print("    ✅ Knowledge base has documents")
                                else:
                                    print("    ⚠️  No documents in knowledge base - use /rag mode to ingest")
                        print("\n💡 The agent will automatically use these tools when needed.")
                        print("   For RAG: Ask questions about your documents and the agent will search them.")
                        print("=" * 70 + "\n")
                        continue
                    
                    elif cmd == 'rag-status' or (cmd == 'status' and mode == 'agent'):
                        # Show RAG status in agent mode
                        has_docs = check_rag_status()
                        print("\n" + "=" * 70)
                        print("📚 RAG Knowledge Base Status")
                        print("=" * 70)
                        if has_docs:
                            try:
                                chunks = rag.retrieve("", top_k=100)  # Get a sample
                                print(f"\n✅ Knowledge Base: Ready")
                                print(f"   Documents available: Yes")
                                print(f"   Sample chunks found: {len(chunks)}")
                                print("\n💡 The agent can search these documents using the 'rag_search' tool.")
                                print("   Just ask questions and the agent will automatically search your documents.\n")
                            except Exception as e:
                                print(f"\n✅ Knowledge base is ready.")
                                print(f"   (Error checking chunks: {e})\n")
                        else:
                            print("\n⚠️  No documents in knowledge base.")
                            print("\n📝 To add documents:")
                            print("   1. Switch to RAG mode: /rag")
                            print("   2. Use: /ingest <file_path>")
                            print("   3. Or exit and use: rag-agent-cli rag ingest --file <file>")
                            print("\n   Once documents are ingested, the agent can search them automatically.\n")
                        print("=" * 70 + "\n")
                        continue
                    
                    elif cmd == 'test-rag' and mode == 'agent':
                        # Test RAG tool directly
                        print("\n🧪 Testing RAG tool directly...\n")
                        test_query = user_input.split(None, 1)[1] if len(user_input.split()) > 1 else "test"
                        try:
                            chunks = rag.retrieve(test_query, top_k=3)
                            if chunks:
                                print(f"✅ RAG tool works! Found {len(chunks)} chunks for query: '{test_query}'\n")
                                for i, chunk in enumerate(chunks[:2], 1):
                                    print(f"Chunk {i}: {chunk.get('content', '')[:200]}...\n")
                                print("💡 The agent should be able to use this tool. Try asking:")
                                print(f"   'Search your documents for {test_query}'")
                                print("   'What information do you have about [topic]?'\n")
                            else:
                                print(f"⚠️  RAG tool works but found no chunks for: '{test_query}'")
                                print("   Try a different query or check if documents are ingested.\n")
                        except Exception as e:
                            print(f"❌ Error testing RAG: {e}\n")
                        continue
                    
                    elif cmd == 'clear':
                        chat_history.clear()
                        print("\n✅ Conversation history cleared\n")
                    
                    elif cmd == 'config' or cmd == 'flags':
                        # Show current configuration
                        print("\n" + "=" * 70)
                        print("⚙️  CURRENT CONFIGURATION")
                        print("=" * 70)
                        print(f"\n📋 Mode: {mode.upper()}")
                        print(f"\n🔧 Agent Settings:")
                        print(f"   RAG Enabled:        ✅ Yes")
                        print(f"   Web Search:         {'✅ Enabled' if args.web_search else '❌ Disabled'}")
                        print(f"   MCP Tools:          {'✅ Enabled' if mcp else '❌ Disabled'}")
                        print(f"   Conversation Memory: {'✅ Enabled' if args.memory else '❌ Disabled'}")
                        print(f"   Debug Mode:         {'✅ Enabled' if debug_logger._get_enabled() else '❌ Disabled'}")
                        if args.system_prompt:
                            print(f"   System Prompt:      ✅ Custom")
                        else:
                            print(f"   System Prompt:      ⚙️  Default")
                        
                        if mode == 'rag':
                            print(f"\n🔧 RAG Settings:")
                            print(f"   LLM Answers:        {'✅ Enabled' if runtime_settings['rag_llm'] else '❌ Disabled (chunks only)'}")
                            print(f"   Top-K Retrieval:    {args.rag_top_k} chunks")
                            print(f"\n💡 Toggle Commands:")
                            print(f"   /toggle-llm        - Toggle LLM answers on/off")
                        
                        # Show available tools
                        if mode == 'agent':
                            tools = agent.get_tools_description()
                            print(f"\n🔧 Available Tools ({len(tools)}):")
                            for tool in tools:
                                print(f"   • {tool['name']}")
                        
                        print("\n💡 All Commands:")
                        if mode == 'rag':
                            print("   /toggle-llm        - Toggle LLM answers")
                            print("   /ingest <file>     - Ingest document")
                            print("   /status            - Check knowledge base")
                        else:
                            print("   /tools             - List available tools")
                            print("   /clear             - Clear conversation history")
                        print("   /config             - Show this configuration")
                        print("   /help               - Show help")
                        print("   /rag or /agent      - Switch modes")
                        print("   /exit               - Exit")
                        print("=" * 70 + "\n")
                        continue
                    
                    else:
                        print(f"\n❌ Unknown command: /{cmd}. Type /help for available commands.\n")
                    
                    continue
                
                # Check for common commands that might be typed without /
                rag_commands = ['ingest', 'query', 'clear', 'export']
                if mode == 'rag' and user_input.lower() in rag_commands:
                    print(f"\n💡 Tip: '{user_input}' is a CLI command, not an interactive command.")
                    print("   In interactive RAG mode, just ask questions about your documents.")
                    print("   Example: 'What is machine learning?'")
                    print("   To use CLI commands, exit and run:")
                    print(f"     rag-agent-cli rag {user_input.lower()} [options]")
                    print("   Type /help for more info.\n")
                    continue
                
                # Check for agent commands typed in wrong mode
                agent_commands = ['invoke', 'tools']
                if mode == 'rag' and user_input.lower().startswith(tuple(agent_commands)):
                    print(f"\n💡 Tip: '{user_input.split()[0]}' is an agent command.")
                    print("   Switch to agent mode with: /agent")
                    print("   Or exit and run: rag-agent-cli agent {command}")
                    print("   Type /help for more info.\n")
                    continue
                
                # Process user input based on mode
                print()  # Blank line before response
                
                if mode == 'agent':
                    # Chat mode - conversational
                    print("🤖 Assistant: ", end="", flush=True)
                    try:
                        # Validate agent is ready
                        if not agent:
                            raise RuntimeError("Agent not initialized properly")
                        
                        response = agent.invoke(
                            user_input,
                            system_prompt=args.system_prompt,
                            chat_history=chat_history if args.memory else None,
                        )
                        
                        # Check if response is valid
                        if response is None:
                            response = "I'm sorry, I didn't get a response. Please try again."
                        elif not isinstance(response, str):
                            response = str(response)
                        
                        print(response)
                        print()  # Extra blank line for readability
                        
                        # Update history if memory is enabled
                        if args.memory:
                            chat_history.append({"role": "user", "content": user_input})
                            chat_history.append({"role": "assistant", "content": response})
                    
                    except KeyboardInterrupt:
                        print("\n\n⚠️  Interrupted. You can continue chatting or type /exit to quit.\n")
                        continue
                    except Exception as e:
                        error_msg = str(e) if str(e) else repr(e)
                        error_type = type(e).__name__
                        print(f"\n❌ Sorry, I encountered an error: {error_type}")
                        if error_msg and error_msg != "NotImplementedError()":
                            print(f"   {error_msg}")
                        else:
                            print(f"   {repr(e)}")
                        
                        # Provide helpful suggestions based on error type
                        if error_type == "NotImplementedError":
                            print("\n💡 This error suggests a compatibility issue with LangChain.")
                            print("   The adapter might be missing a required method.")
                            print("   Try running with --verbose to see full details.")
                        elif "api" in error_msg.lower() or "key" in error_msg.lower():
                            print("\n💡 Tip: Check your API key in .env file (OPENAI_API_KEY)")
                        elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                            print("\n💡 Tip: Check your internet connection")
                        elif "rag" in error_msg.lower() or "document" in error_msg.lower():
                            print("\n💡 Tip: Make sure you have documents ingested in RAG mode")
                        
                        # Always show traceback for NotImplementedError to help debug
                        if error_type == "NotImplementedError" or debug_logger._get_enabled():
                            import traceback
                            print("\n📋 Full traceback:")
                            traceback.print_exc()
                        print()
                
                else:  # RAG workflow mode
                    # Check if we have documents first
                    has_docs = check_rag_status()
                    if not has_docs:
                        print("⚠️  No documents in knowledge base yet.")
                        print("   Use /ingest <file_path> to add documents, or exit and use:")
                        print("   rag-agent-cli rag ingest --file <file>\n")
                        continue
                    
                    # RAG query workflow
                    print("🔍 Searching your knowledge base...\n")
                    try:
                        if runtime_settings['rag_llm']:
                            print("💭 Generating answer with LLM...\n")
                            answer = rag.query(user_input, llm_client=llm, top_k=args.rag_top_k)
                            print(f"📚 Answer:\n{answer}\n")
                            print("💡 Tip: Use /toggle-llm to switch to showing document chunks instead.\n")
                        else:
                            chunks = rag.retrieve(user_input, top_k=args.rag_top_k)
                            if chunks:
                                print(f"✅ Found {len(chunks)} relevant document(s):\n")
                                for i, chunk in enumerate(chunks, 1):
                                    content = chunk.get('content', '')
                                    print(f"📄 Document {i}:")
                                    print(f"   {content[:400]}{'...' if len(content) > 400 else ''}")
                                    if chunk.get('metadata'):
                                        source = chunk['metadata'].get('source', 'Unknown')
                                        print(f"   📍 Source: {source}")
                                    print()
                                
                                print("💡 Tip: Use /toggle-llm to get LLM-generated answers instead of chunks.")
                                print("   Or type /help to see all available commands.\n")
                            else:
                                print("❌ No relevant documents found.")
                                print("   Try rephrasing your query or ingesting more documents.\n")
                    
                    except KeyboardInterrupt:
                        print("\n\n⚠️  Interrupted. You can continue querying or type /exit to quit.\n")
                        continue
                    except Exception as e:
                        error_msg = str(e) if str(e) else repr(e)
                        error_type = type(e).__name__
                        print(f"❌ Error querying: {error_type}")
                        if error_msg:
                            print(f"   {error_msg}")
                        else:
                            print(f"   {repr(e)}")
                        
                        if debug_logger._get_enabled():
                            import traceback
                            print("\n📋 Full traceback:")
                            traceback.print_exc()
                        print()
            
            except EOFError:
                # Handle Ctrl+D
                print("\n\n👋 Goodbye!")
                break
            
            except KeyboardInterrupt:
                # Handle Ctrl+C
                print("\n\n👋 Goodbye!")
                break
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}", file=sys.stderr)
        if debug_logger._get_enabled():
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Unified RAG and Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended for conversations)
  %(prog)s interactive
  %(prog)s i --mode rag  # Start in RAG mode
  %(prog)s i --no-web-search  # Disable web search
  
  # RAG operations
  %(prog)s rag ingest --file document.txt
  %(prog)s rag query "What is in the documents?" --top-k 5
  %(prog)s rag query "Question?" --llm
  %(prog)s rag clear
  %(prog)s rag export --output corpus.jsonl --format jsonl
  
  # Agent operations
  %(prog)s agent invoke --message "What documents have you ingested?"
  %(prog)s agent invoke --file question.txt --output answer.txt --web-search
  %(prog)s agent tools
  %(prog)s agent tools --json
        """
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Mode: rag, agent, or interactive")
    
    # Interactive mode (top-level subcommand)
    interactive_parser = subparsers.add_parser("interactive", aliases=["i"], help="Start interactive mode")
    interactive_parser.add_argument("--system-prompt", "-s", help="System prompt for agent")
    interactive_parser.add_argument("--web-search", action="store_true", default=True, help="Enable web search tool (default: True)")
    interactive_parser.add_argument("--no-web-search", dest="web_search", action="store_false", help="Disable web search tool")
    interactive_parser.add_argument("--mcp", action="store_true", help="Enable MCP tools")
    interactive_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    interactive_parser.add_argument("--memory", action="store_true", default=True, help="Enable conversation memory (default: True)")
    interactive_parser.add_argument("--no-memory", dest="memory", action="store_false", help="Disable conversation memory")
    interactive_parser.add_argument("--mode", dest="initial_mode", choices=["agent", "rag"], default="agent", help="Initial mode (default: agent)")
    interactive_parser.add_argument("--rag-llm", action="store_true", help="Use LLM for RAG queries in RAG mode")
    interactive_parser.add_argument("--rag-top-k", "-k", type=int, default=4, help="Top-k for RAG retrieval (default: 4)")
    
    # RAG subcommands
    rag_parser = subparsers.add_parser("rag", help="RAG operations")
    rag_subparsers = rag_parser.add_subparsers(dest="command", help="RAG command")
    
    # RAG ingest
    ingest_parser = rag_subparsers.add_parser("ingest", help="Ingest documents into RAG")
    ingest_group = ingest_parser.add_mutually_exclusive_group(required=True)
    ingest_group.add_argument("--file", "-f", type=Path, help="File to ingest")
    ingest_group.add_argument("--text", "-t", help="Text to ingest")
    ingest_parser.add_argument("--metadata", "-m", help="Metadata as JSON")
    
    # RAG query
    query_parser = rag_subparsers.add_parser("query", help="Query RAG")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--top-k", "-k", type=int, default=4, help="Number of chunks")
    query_parser.add_argument("--llm", action="store_true", help="Use LLM for answer generation")
    
    # RAG clear
    rag_subparsers.add_parser("clear", help="Clear RAG store")
    
    # RAG export
    export_parser = rag_subparsers.add_parser("export", help="Export RAG corpus")
    export_parser.add_argument("--output", "-o", type=Path, help="Output file")
    export_parser.add_argument("--format", "-f", choices=["json", "jsonl"], default="json", help="Export format")
    
    # Agent subcommands
    agent_parser = subparsers.add_parser("agent", help="Agent operations")
    agent_subparsers = agent_parser.add_subparsers(dest="command", help="Agent command")
    
    # Agent invoke
    invoke_parser = agent_subparsers.add_parser("invoke", help="Invoke agent")
    invoke_group = invoke_parser.add_mutually_exclusive_group(required=True)
    invoke_group.add_argument("--message", "-m", help="Message to send")
    invoke_group.add_argument("--file", "-f", type=Path, help="File containing message")
    invoke_parser.add_argument("--system-prompt", "-s", help="System prompt")
    invoke_parser.add_argument("--output", "-o", type=Path, help="Output file")
    invoke_parser.add_argument("--web-search", action="store_true", default=True, help="Enable web search tool (default: True)")
    invoke_parser.add_argument("--no-web-search", dest="web_search", action="store_false", help="Disable web search tool")
    invoke_parser.add_argument("--mcp", action="store_true", help="Enable MCP tools")
    invoke_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # Agent tools
    tools_parser = agent_subparsers.add_parser("tools", help="List agent tools")
    tools_parser.add_argument("--json", action="store_true", help="Output as JSON")
    tools_parser.add_argument("--web-search", action="store_true", default=True, help="Enable web search tool (default: True)")
    tools_parser.add_argument("--no-web-search", dest="web_search", action="store_false", help="Disable web search tool")
    tools_parser.add_argument("--mcp", action="store_true", help="Enable MCP tools")
    
    args = parser.parse_args()
    
    # Handle interactive mode
    if args.mode in ("interactive", "i"):
        try:
            return interactive_command(args)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            if hasattr(args, 'verbose') and args.verbose:
                traceback.print_exc()
            return 1
    
    if not args.mode:
        parser.print_help()
        return 1
    
    try:
        # RAG commands
        if args.mode == "rag":
            if args.command == "ingest":
                return ingest_command(args)
            elif args.command == "query":
                return query_command(args)
            elif args.command == "clear":
                return clear_command(args)
            elif args.command == "export":
                return export_command(args)
            else:
                rag_parser.print_help()
                return 1
        
        # Agent commands
        elif args.mode == "agent":
            if args.command == "invoke":
                return agent_invoke_command(args)
            elif args.command == "tools":
                return agent_tools_command(args)
            else:
                agent_parser.print_help()
                return 1
        
        else:
            parser.print_help()
            return 1
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        # Check config DEBUG flag or verbose flag
        settings = get_settings()
        if (hasattr(args, 'verbose') and args.verbose) or settings.DEBUG:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
