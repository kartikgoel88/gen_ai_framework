"""CLI tool for agent operations.

This CLI uses FastAPI dependencies (get_agent) for consistency with the API.
For standalone Python scripts, consider using create_tool_agent() instead:
    from src.framework.agents import create_tool_agent
    from src.framework.api.deps import get_llm, get_rag
    
    llm = get_llm(settings)
    rag = get_rag(settings)
    agent = create_tool_agent(llm=llm, rag_client=rag, enable_web_search=True)
"""

import argparse
import json
import sys
from pathlib import Path

from src.framework.config import get_settings
from src.framework.api.deps import get_agent, get_rag, get_mcp_client


def invoke_command(args):
    """Invoke agent."""
    settings = get_settings()
    rag = get_rag(settings)
    mcp = get_mcp_client(settings)
    
    agent = get_agent(settings, rag=rag, mcp=mcp)
    
    if args.file:
        message = Path(args.file).read_text()
    else:
        message = args.message
    
    system_prompt = None
    if args.system_prompt:
        system_prompt = args.system_prompt
    
    response = agent.invoke(message, system_prompt=system_prompt)
    
    if args.output:
        Path(args.output).write_text(response)
        print(f"✅ Response written to {args.output}")
    else:
        print(response)
    
    return 0


def tools_command(args):
    """List agent tools."""
    settings = get_settings()
    rag = get_rag(settings)
    mcp = get_mcp_client(settings)
    
    agent = get_agent(settings, rag=rag, mcp=mcp)
    tools = agent.get_tools_description()
    
    if args.json:
        print(json.dumps(tools, indent=2))
    else:
        print("Available Tools:")
        for tool in tools:
            print(f"\n  {tool['name']}")
            print(f"    {tool['description']}")
    
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Agent operations CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Invoke command
    invoke_parser = subparsers.add_parser("invoke", help="Invoke agent")
    invoke_group = invoke_parser.add_mutually_exclusive_group(required=True)
    invoke_group.add_argument("--message", "-m", help="Message to send")
    invoke_group.add_argument("--file", "-f", type=Path, help="File containing message")
    invoke_parser.add_argument("--system-prompt", "-s", help="System prompt")
    invoke_parser.add_argument("--output", "-o", type=Path, help="Output file")
    
    # Tools command
    tools_parser = subparsers.add_parser("tools", help="List agent tools")
    tools_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == "invoke":
            return invoke_command(args)
        elif args.command == "tools":
            return tools_command(args)
        else:
            parser.print_help()
            return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
