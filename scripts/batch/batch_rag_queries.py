#!/usr/bin/env python3
"""
Batch script: run a list of queries through RAG and write answers to JSONL.

Directly invokes the framework RAG + LLM; no HTTP/FastAPI.
Idempotent: skips query IDs already present in output file (resume).

Input JSONL format: one object per line with "id" and "query" (or "question").
  {"id": "1", "query": "What is X?"}
  {"id": "2", "question": "Explain Y."}

Output JSONL: {"id", "query", "answer", "error?"}

Usage:
  python -m scripts.batch.batch_rag_queries --input queries.jsonl --output answers.jsonl
  python -m scripts.batch.batch_rag_queries --input q.jsonl -o out.jsonl --model gpt-4o --retries 3
"""

import json
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.batch.base import (
    BaseBatchRunner,
    ExitCode,
    append_jsonl,
    existing_output_ids,
    load_jsonl,
    retry_with_backoff,
)
from scripts.batch.common import apply_framework_overrides


def _load_queries(path: Path) -> list[dict]:
    """Load queries from JSONL. Each line: {id, query} or {id, question}."""
    rows = load_jsonl(path)
    out: list[dict] = []
    for i, row in enumerate(rows):
        qid = row.get("id")
        if qid is None:
            qid = str(i + 1)
        query = row.get("query") or row.get("question") or ""
        if not query.strip():
            continue
        out.append({"id": str(qid), "query": query.strip()})
    return out


class RAGQueriesBatchRunner(BaseBatchRunner):
    """Batch runner: RAG query over a list of questions; write answers to JSONL."""

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument("--input", "-i", type=Path, required=True, help="Input JSONL: one {id, query} per line.")
        parser.add_argument("--output", "-o", type=Path, required=True, help="Output JSONL: one {id, query, answer} per line.")
        parser.add_argument("--model", "-m", type=str, default=None, help="Override LLM model.")
        parser.add_argument("--retries", type=int, default=3, metavar="N", help="Max retries per query.")
        parser.add_argument("--top-k", type=int, default=4, help="RAG retrieval top_k (default: 4).")
        parser.add_argument("--dry-run", action="store_true", help="List queries that would be run (and already done if output exists).")

    def parse_args(self):
        import argparse
        p = argparse.ArgumentParser(description=self.description, formatter_class=argparse.RawDescriptionHelpFormatter)
        self.add_arguments(p)
        return p.parse_args()

    def validate_args(self, args):
        if not args.input.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        args.output.parent.mkdir(parents=True, exist_ok=True)

    def dry_run(self, args):
        queries = _load_queries(args.input)
        done = existing_output_ids(args.output, id_key="id")
        to_do = [q for q in queries if q["id"] not in done]
        self.log.info("Dry run: total queries %s, already in output %s, to run %s", len(queries), len(done), len(to_do))
        for q in to_do[:15]:
            self.log.info("  id=%s query=%s...", q["id"], (q["query"][:50] + "..." if len(q["query"]) > 50 else q["query"]))
        if len(to_do) > 15:
            self.log.info("  ... and %s more", len(to_do) - 15)

    def run_batch(self, args):
        apply_framework_overrides(model=args.model)

        from src.framework.config import get_settings
        from src.framework.api.deps import get_rag, get_llm

        settings = get_settings()
        rag = get_rag(settings)
        llm = get_llm(settings)

        queries = _load_queries(args.input)
        done_ids = existing_output_ids(args.output, id_key="id")
        to_run = [q for q in queries if q["id"] not in done_ids]
        self.log.info("Total queries: %s, already in output: %s, to run: %s", len(queries), len(done_ids), len(to_run))

        success_count = 0
        failure_count = 0
        errors: list[dict] = []

        for q in to_run:
            qid = q["id"]
            query = q["query"]
            try:
                def do_query():
                    return rag.query(query, llm_client=llm, top_k=args.top_k)

                answer = retry_with_backoff(
                    do_query,
                    max_attempts=args.retries,
                    log=self.log,
                    is_retryable=lambda e: "rate limit" in str(e).lower() or "429" in str(e),
                )
                record = {"id": qid, "query": query, "answer": answer}
                append_jsonl(args.output, record)
                success_count += 1
            except Exception as e:
                self.log.exception("Query id=%s failed: %s", qid, e)
                failure_count += 1
                errors.append({"id": qid, "query": query, "error": str(e)})
                append_jsonl(args.output, {"id": qid, "query": query, "answer": None, "error": str(e)})

        return {
            "success_count": success_count,
            "failure_count": failure_count,
            "errors": errors,
            "result": {"answered": success_count, "failed": failure_count, "skipped": len(done_ids)},
        }


def main() -> int:
    runner = RAGQueriesBatchRunner(
        description="Batch RAG queries: run a list of questions through RAG and write answers to JSONL (direct framework invocation).",
    )
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
