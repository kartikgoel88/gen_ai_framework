"""
Parse passport images (PassportEye MRZ by default); optionally extract entities via MRZ + LLM.

Usage:
  python -m src.clients.documents.parse_passport [--images-dir data/images]
  python -m src.clients.documents.parse_passport --extract [--output results.json]

--method: passporteye (default), pytesseract, docling.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from src.framework.documents import IMAGE_EXTENSIONS, OcrProcessor, PassportEyeProcessor
from src.framework.documents.types import ExtractResult
from src.framework.documents.base import BaseDocumentProcessor
from src.framework.utils.json_utils import parse_json_from_text
from src.framework.utils.date_utils import normalize_date

PASSPORT_ENTITIES = (
    "passport_number",
    "name",
    "date_of_birth",
    "date_of_issue",
    "date_of_expiry",
)
_NULL_ENTITIES = {k: None for k in PASSPORT_ENTITIES}


def _normalize_ocr_text(text: str) -> str:
    """Collapse space-separated single characters so 'V 4 1 3 4 1 0 2' -> 'V4134102'. Helps when OCR outputs one char per token."""
    if not text or not text.strip():
        return text
    tokens = text.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        run = []
        while i < len(tokens) and len(tokens[i]) == 1:
            run.append(tokens[i])
            i += 1
        if run:
            out.append("".join(run))
        if i < len(tokens):
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def _entities_from_mrz(metadata: dict) -> dict[str, Any]:
    """Build PASSPORT_ENTITIES from PassportEye MRZ metadata (no LLM)."""
    mrz = metadata.get("mrz") or {}
    if not mrz:
        return _NULL_ENTITIES.copy()
    surname = (mrz.get("surname") or "").strip()
    names = (mrz.get("names") or "").strip()
    name = f"{names} {surname}".strip() if (names or surname) else None
    doc_num = (mrz.get("number") or mrz.get("personal_number") or "").strip() or None
    return {
        "passport_number": doc_num,
        "name": name or None,
        "date_of_birth": normalize_date((mrz.get("date_of_birth") or "").strip() or None),
        "date_of_issue": None,  # MRZ does not include issue date
        "date_of_expiry": normalize_date((mrz.get("expiration_date") or "").strip() or None),
    }


def extract_passport_entities(ocr_text: str, llm: Any, debug: bool = False) -> dict[str, Any]:
    """Extract passport entities from OCR text via LLM."""
    if not (ocr_text or "").strip():
        return _NULL_ENTITIES.copy()
    text = _normalize_ocr_text(ocr_text)
    if debug and text != ocr_text:
        print("[debug] OCR normalized (space-collapsed) for LLM", file=sys.stderr)
    prompt = f"""Extract passport fields from this text. Reply with ONLY a JSON object.
Keys: passport_number, name, date_of_birth, date_of_issue, date_of_expiry. Use null if not found.
CRITICAL: Every date MUST be exactly in format YYYY-MM-DD (e.g. 1990-01-15, 2030-12-31). No other format.
Example: {{"passport_number": "AB123456", "name": "John Doe", "date_of_birth": "1985-06-20", "date_of_issue": "2020-03-10", "date_of_expiry": "2030-03-09"}}

Text:
{text[:5000]}"""
    try:
        result = llm.invoke_structured(prompt)
        if debug and isinstance(result, dict):
            print(f"[debug] LLM result keys: {list(result.keys())}", file=sys.stderr)
        out = {k: result.get(k) for k in PASSPORT_ENTITIES} if isinstance(result, dict) else _NULL_ENTITIES.copy()
        if all(out.get(k) is None for k in PASSPORT_ENTITIES) and isinstance(result, dict) and result.get("raw"):
            parsed = parse_json_from_text(str(result["raw"]))
            if parsed:
                out = {k: parsed.get(k) for k in PASSPORT_ENTITIES}
        # Force all date fields to YYYY-MM-DD or null
        for key in ("date_of_birth", "date_of_issue", "date_of_expiry"):
            raw = out.get(key)
            out[key] = normalize_date(raw) if raw is not None and raw != "" else None
        return out
    except Exception as e:
        if debug:
            print(f"[debug] extract error: {e}", file=sys.stderr)
        return _NULL_ENTITIES.copy()


def _processor_for_method(method: str) -> BaseDocumentProcessor:
    if method == "passporteye":
        return PassportEyeProcessor()
    if method == "docling":
        from src.framework.docling.processor import DoclingProcessor
        return DoclingProcessor(default_export_format="text")
    return OcrProcessor(ocr_backend="pytesseract")


def parse_passport_images(
    images_dir: Path,
    ocr_processor: BaseDocumentProcessor | None = None,
    method: str = "passporteye",
) -> list[dict]:
    """Run OCR on images in images_dir. Returns list of {file, path, text, metadata, error}."""
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        return [{"file": str(images_dir), "text": "", "metadata": {}, "error": "Not a directory"}]

    processor = ocr_processor or _processor_for_method(method)
    results = []
    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        r: ExtractResult = processor.extract(path)
        results.append({
            "file": path.name,
            "path": str(path),
            "text": r.text,
            "metadata": r.metadata,
            "error": r.error,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR passport images; optional entity extraction.")
    parser.add_argument("-dir", "--images-dir", type=Path, default=Path(__file__).resolve().parents[3] / "data" / "images")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--method", choices=("passporteye", "pytesseract", "docling"), default="passporteye",
                        help="OCR method: passporteye (default, MRZ), pytesseract (Tesseract), or docling (layout-aware)")
    parser.add_argument("--extract", action="store_true", help="Extract entities via LLM (use LLM_PROVIDER=local for PII-safe)")
    parser.add_argument("--debug", action="store_true", help="Print OCR excerpt and LLM response when extracting")
    args = parser.parse_args()

    if not args.quiet:
        print(f"OCR method: {args.method}", file=sys.stderr)
    results = parse_passport_images(args.images_dir, method=args.method)

    if args.extract:
        from src.framework.config import get_settings
        from src.framework.api.deps_llm import get_llm
        settings = get_settings()
        provider = (settings.LLM_PROVIDER or "openai").strip().lower()
        model = (getattr(settings, "LLM_LOCAL_MODEL", None) or settings.LLM_MODEL) if provider == "local" else settings.LLM_MODEL
        print(f"Extracting entities with LLM: provider={provider}, model={model}", file=sys.stderr)
        llm = get_llm(settings)
        for r in results:
            text = (r.get("text") or "").strip()
            if args.debug and text:
                print(f"[debug] {r.get('file')} OCR length={len(text)}", file=sys.stderr)
            mrz = (r.get("metadata") or {}).get("mrz")
            if mrz and isinstance(mrz, dict):
                r["entities"] = _entities_from_mrz(r["metadata"])
                if text and any(r["entities"].get(k) is None for k in PASSPORT_ENTITIES):
                    llm_entities = extract_passport_entities(text, llm, debug=args.debug)
                    for k in PASSPORT_ENTITIES:
                        if r["entities"].get(k) is None and llm_entities.get(k) is not None:
                            r["entities"][k] = llm_entities[k]
            else:
                r["entities"] = extract_passport_entities(text, llm, debug=args.debug) if text else _NULL_ENTITIES.copy()

    if args.output:
        args.output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {len(results)} results to {args.output}", file=sys.stderr)

    if not args.quiet:
        for r in results:
            print(f"\n--- {r['file']} ---")
            if r.get("error"):
                print(f"Error: {r['error']}")
            else:
                if args.extract:
                    for k in PASSPORT_ENTITIES:
                        print(f"  {k}: {(r.get('entities') or {}).get(k)}")
                raw = (r.get("text") or "").strip()
                print("(no OCR text)" if not raw else raw)
                if not raw or (args.extract and all((r.get("entities") or {}).get(k) is None for k in PASSPORT_ENTITIES)):
                    print("  → Use a clear front-page passport image.", file=sys.stderr)

    if not results:
        print("No image files found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
