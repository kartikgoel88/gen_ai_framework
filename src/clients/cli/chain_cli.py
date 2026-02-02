"""CLI tool for chain operations."""

import argparse
import json
import sys
from pathlib import Path

from src.framework.config import get_settings
from src.framework.api.deps import get_llm, get_rag, get_rag_chain
from src.framework.chains import (
    PromptChain,
    StructuredChain,
    SummarizationChain,
    ClassificationChain,
    ExtractionChain
)


def invoke_chain(chain_type: str, inputs: dict, settings, llm, rag):
    """Invoke a chain."""
    if chain_type == "rag":
        chain = get_rag_chain(llm=llm, rag=rag)
        return chain.invoke(inputs)
    elif chain_type == "prompt":
        template = inputs.pop("template", "{prompt}")
        chain = PromptChain(llm=llm, template=template)
        return chain.invoke(inputs)
    elif chain_type == "structured":
        template = inputs.pop("template", "{prompt}")
        chain = StructuredChain(llm=llm, template=template)
        return chain.invoke(inputs)
    elif chain_type == "summarization":
        chain = SummarizationChain(llm=llm)
        return chain.invoke(inputs)
    elif chain_type == "classification":
        labels = inputs.pop("labels", "positive,negative,neutral").split(",")
        chain = ClassificationChain(llm=llm, labels=[l.strip() for l in labels])
        return chain.invoke(inputs)
    elif chain_type == "extraction":
        schema = inputs.pop("schema", "key facts and entities")
        chain = ExtractionChain(llm=llm, schema=schema)
        return chain.invoke(inputs)
    else:
        raise ValueError(f"Unknown chain type: {chain_type}")


def invoke_command(args):
    """Invoke a chain."""
    settings = get_settings()
    llm = get_llm(settings)
    rag = get_rag(settings)
    
    # Parse inputs
    if args.inputs_file:
        inputs = json.loads(Path(args.inputs_file).read_text())
    else:
        inputs = json.loads(args.inputs_json)
    
    # Invoke chain
    result = invoke_chain(args.chain_type, inputs, settings, llm, rag)
    
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"✅ Result written to {args.output}")
    else:
        print(json.dumps(result, indent=2))
    
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chain operations CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Invoke command
    invoke_parser = subparsers.add_parser("invoke", help="Invoke a chain")
    invoke_parser.add_argument(
        "chain_type",
        choices=["rag", "prompt", "structured", "summarization", "classification", "extraction"],
        help="Chain type"
    )
    invoke_group = invoke_parser.add_mutually_exclusive_group(required=True)
    invoke_group.add_argument("--inputs-json", "-j", help="Inputs as JSON string")
    invoke_group.add_argument("--inputs-file", "-f", type=Path, help="Inputs JSON file")
    invoke_parser.add_argument("--output", "-o", type=Path, help="Output file")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == "invoke":
            return invoke_command(args)
        else:
            parser.print_help()
            return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
