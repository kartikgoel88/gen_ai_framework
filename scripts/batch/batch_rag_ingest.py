#!/usr/bin/env python3
"""
Batch script: ingest a directory of documents into RAG (vector store).

Directly invokes the framework RAG client; no HTTP/FastAPI.
Idempotent: uses an optional manifest file to skip already-ingested paths (resume).

Usage:
  python -m scripts.batch.batch_rag_ingest --input ./docs --manifest ingested.jsonl
  python -m scripts.batch.batch_rag_ingest --input ./docs --persist-dir ./data/chroma_db --model gpt-4o --retries 3
"""

import json
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.batch.base import BaseBatchRunner, ExitCode, retry_with_backoff
from scripts.batch.common import apply_framework_overrides

# Extensions we can extract text from (same as RAG CLI / document processor)
INGEST_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".html", ".htm"}


class RAGIngestBatchRunner(BaseBatchRunner):
    """Batch runner: ingest documents from a directory into RAG."""

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument("--input", "-i", type=Path, required=True, help="Directory of documents to ingest.")
        parser.add_argument(
            "--manifest",
            type=Path,
            default=None,
            help="JSONL file of already-ingested paths (one path per line in 'path' key). Used for resume/idempotency.",
        )
        parser.add_argument(
            "--persist-dir",
            type=Path,
            default=None,
            help="Chroma persist directory (default: config CHROMA_PERSIST_DIR).",
        )
        parser.add_argument(
            "--extensions",
            type=str,
            default=".txt,.md,.pdf,.docx,.html,.htm",
            help="Comma-separated file extensions to ingest (default: .txt,.md,.pdf,.docx,.html,.htm).",
        )
        parser.add_argument("--retries", type=int, default=3, metavar="N", help="Max retries per document.")
        parser.add_argument("--dry-run", action="store_true", help="List files that would be ingested.")

    def parse_args(self):
        import argparse
        p = argparse.ArgumentParser(description=self.description, formatter_class=argparse.RawDescriptionHelpFormatter)
        self.add_arguments(p)
        return p.parse_args()

    def validate_args(self, args):
        if not args.input.is_dir():
            raise NotADirectoryError(f"Input is not a directory: {args.input}")
        if args.persist_dir is not None:
            args.persist_dir.mkdir(parents=True, exist_ok=True)

    def dry_run(self, args):
        to_ingest = self._collect_files(args)
        self.log.info("Dry run: would ingest %s files from %s", len(to_ingest), args.input)
        for p in to_ingest[:20]:
            self.log.info("  %s", p.relative_to(args.input))
        if len(to_ingest) > 20:
            self.log.info("  ... and %s more", len(to_ingest) - 20)

    def _collect_files(self, args):
        exts = {e.strip().lower() for e in args.extensions.split(",") if e.strip()}
        exts = exts or {".txt", ".md"}
        paths: list[Path] = []
        for f in args.input.rglob("*"):
            if f.is_file() and f.suffix.lower() in exts:
                paths.append(f)
        return sorted(paths)

    def _already_ingested(self, manifest_path: Path | None) -> set[str]:
        if not manifest_path or not manifest_path.exists():
            return set()
        seen: set[str] = set()
        with manifest_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "path" in obj:
                        seen.add(str(Path(obj["path"]).resolve()))
                except json.JSONDecodeError:
                    continue
        return seen

    def _append_manifest(self, manifest_path: Path, path: Path) -> None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"path": str(path.resolve())}, ensure_ascii=False) + "\n")

    def run_batch(self, args):
        if args.persist_dir is not None:
            apply_framework_overrides(env_overrides={"CHROMA_PERSIST_DIR": str(args.persist_dir)})
        else:
            apply_framework_overrides()

        from src.framework.config import get_settings
        from src.framework.api.deps import get_rag
        from src.framework.documents.processor import DocumentProcessor
        from src.framework.documents import OcrProcessor

        settings = get_settings()
        rag = get_rag(settings)
        ocr = OcrProcessor(pdf_dpi=300, pdf_min_text_len=10)
        doc_processor = DocumentProcessor()

        to_ingest = self._collect_files(args)
        already = self._already_ingested(args.manifest)
        to_process = [p for p in to_ingest if str(p.resolve()) not in already]
        self.log.info("Total files: %s, already in manifest: %s, to process: %s", len(to_ingest), len(already), len(to_process))

        success_count = 0
        failure_count = 0
        errors: list[dict] = []

        for path in to_process:
            try:
                result = doc_processor.extract(path)
                text_str = (result.text or "").strip() if hasattr(result, "text") else ""
                if result.error:
                    self.log.warning("Extract error for %s: %s", path, result.error)
                if not text_str:
                    failure_count += 1
                    errors.append({"path": str(path), "error": result.error or "No text extracted"})
                    continue

                def add_doc():
                    rag.add_documents(
                        [text_str],
                        metadatas=[{"source": str(path), "path": str(path.resolve())}],
                    )

                retry_with_backoff(
                    add_doc,
                    max_attempts=args.retries,
                    log=self.log,
                    is_retryable=lambda e: "rate limit" in str(e).lower() or "429" in str(e),
                )
                success_count += 1
                if args.manifest:
                    self._append_manifest(args.manifest, path)
            except Exception as e:
                self.log.exception("Failed to ingest %s: %s", path, e)
                failure_count += 1
                errors.append({"path": str(path), "error": str(e)})

        return {
            "success_count": success_count,
            "failure_count": failure_count,
            "errors": errors,
            "result": {"ingested": success_count, "skipped": len(already), "failed": failure_count},
        }


def main() -> int:
    runner = RAGIngestBatchRunner(
        description="Batch RAG ingest: load documents from a directory into the vector store (direct framework invocation).",
    )
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
