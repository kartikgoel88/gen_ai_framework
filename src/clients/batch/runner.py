"""
Batch runner: run folder-based bill processing from the backend (CLI).
Uses the same policy-once, per-employee-folder flow as POST /batch/process-folders.

Usage:
  python -m src.clients.batch.runner --policy path/to/policy.pdf --folders path/to/employee_folders/
  python -m src.clients.batch.runner --policy path/to/policy.txt --zip path/to/folders.zip --output results.json
  python -m src.clients.batch.runner --policy policy.pdf --folders ./data/bills --client-addresses clients.json
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

# Ensure project root on path when run as __main__
if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[3]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from src.framework.config import get_settings
from src.framework.llm.base import LLMClient
from src.framework.documents.processor import DocumentProcessor
from src.framework.documents import OcrProcessor
from src.framework.api.deps_llm import get_llm

from .service import BatchExpenseService


def _create_llm_from_settings(settings) -> LLMClient:
    """Build LLM client from framework settings (no FastAPI)."""
    return get_llm(settings)


def _load_policy_text(policy_path: Path, batch_service: BatchExpenseService) -> str:
    """Load policy text from a file (PDF/TXT/DOCX)."""
    path = Path(policy_path)
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")
    text = batch_service.extract_text_from_file(path)
    return (text or "").strip() or "Default: approve if amount is reasonable; reject if amount missing or suspicious."


def _get_folder_paths(
    folders_dir: Path | None,
    zip_path: Path | None,
    extract_root: Path | None = None,
) -> list[Path]:
    """Return list of paths to employee folders: from --folders dir or from --zip (extract to extract_root)."""
    if folders_dir is not None:
        folders_dir = Path(folders_dir)
        if not folders_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {folders_dir}")
        return [folders_dir / d.name for d in folders_dir.iterdir() if d.is_dir()]
    if zip_path is not None:
        zip_path = Path(zip_path)
        if not zip_path.is_file():
            raise FileNotFoundError(f"ZIP not found: {zip_path}")
        base = Path(extract_root) if extract_root else Path("./uploads/batch_runner_extract")
        root = base / zip_path.stem
        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)
        return [root / d.name for d in root.iterdir() if d.is_dir()]
    return []


def _load_client_addresses(path: Path | None) -> dict:
    """Load client addresses JSON from file."""
    if not path or not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        out = json.load(f)
    return out if isinstance(out, dict) else {}


def run(
    policy_path: Path,
    folders_dir: Path | None = None,
    zip_path: Path | None = None,
    client_addresses_path: Path | None = None,
    output_path: Path | None = None,
    upload_dir: str | Path = "./uploads",
) -> dict:
    """
    Run batch processing: policy once, one folder per employee.
    Returns the same structure as process_folders (folders + summary).
    """
    settings = get_settings()
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    ocr = OcrProcessor(pdf_dpi=300, pdf_min_text_len=10)
    doc_processor = DocumentProcessor(upload_dir=str(upload_dir), pdf_processor=ocr)
    ocr_processor = ocr
    llm = _create_llm_from_settings(settings)
    batch_service = BatchExpenseService(
        llm=llm,
        doc_processor=doc_processor,
        ocr_processor=ocr_processor,
    )

    policy_text = _load_policy_text(Path(policy_path), batch_service)
    extract_root = upload_dir / "batch_runner_extract" if zip_path else None
    folder_paths = _get_folder_paths(
        folders_dir and Path(folders_dir),
        zip_path and Path(zip_path) if zip_path else None,
        extract_root=extract_root,
    )
    if not folder_paths:
        return {"folders": [], "summary": {"approved": 0, "rejected": 0, "total": 0}}

    client_addresses = _load_client_addresses(client_addresses_path and Path(client_addresses_path))
    out = batch_service.process_folders(
        folder_paths,
        policy_text=policy_text,
        client_addresses=client_addresses,
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run batch bill processing: employee folders + policy (once).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--policy", "-p", required=True, type=Path, help="Path to policy document (PDF/TXT/DOCX).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--folders", "-f", type=Path, help="Path to directory containing one subfolder per employee (folder name = emp_id_emp_name_month_client).")
    group.add_argument("--zip", "-z", type=Path, help="Path to ZIP whose top-level entries are employee folders.")
    parser.add_argument("--client-addresses", "-c", type=Path, help="Path to JSON file: {\"TESCO\": [\"addr1\"], \"AMEX\": [\"addr2\"]} for cab address validation.")
    parser.add_argument("--output", "-o", type=Path, help="Write results JSON to this file; default stdout.")
    parser.add_argument("--upload-dir", type=Path, default=Path("./uploads"), help="Upload/temp directory (default: ./uploads).")
    args = parser.parse_args()

    try:
        result = run(
            policy_path=args.policy,
            folders_dir=args.folders,
            zip_path=args.zip,
            client_addresses_path=args.client_addresses,
            output_path=args.output,
            upload_dir=args.upload_dir,
        )
        payload = json.dumps(result, indent=2, default=str, ensure_ascii=False)
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
            print(f"Wrote {args.output}", file=sys.stderr)
        else:
            print(payload)
        return 0
    except (ValueError, FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
