"""CLI tool for RAG operations."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from src.framework.config import get_settings
from src.framework.api.deps import get_rag, get_llm


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


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG operations CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into RAG")
    ingest_group = ingest_parser.add_mutually_exclusive_group(required=True)
    ingest_group.add_argument("--file", "-f", type=Path, help="File to ingest")
    ingest_group.add_argument("--text", "-t", help="Text to ingest")
    ingest_parser.add_argument("--metadata", "-m", help="Metadata as JSON")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query RAG")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--top-k", "-k", type=int, default=4, help="Number of chunks")
    query_parser.add_argument("--llm", action="store_true", help="Use LLM for answer generation")
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear RAG store")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export RAG corpus")
    export_parser.add_argument("--output", "-o", type=Path, help="Output file")
    export_parser.add_argument("--format", "-f", choices=["json", "jsonl"], default="json", help="Export format")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == "ingest":
            return ingest_command(args)
        elif args.command == "query":
            return query_command(args)
        elif args.command == "clear":
            return clear_command(args)
        elif args.command == "export":
            return export_command(args)
        else:
            parser.print_help()
            return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
