"""PassportEye processor: MRZ extraction from passport/ID images (OpenCV + pytesseract MRZ mode)."""

import warnings
from pathlib import Path
from typing import Any, Union

from .base import BaseDocumentProcessor
from .types import IMAGE_EXTENSIONS, ExtractResult


def _mrz_to_text_and_metadata(mrz: Any) -> tuple[str, dict[str, Any]]:
    """Turn PassportEye MRZ result into plain text and metadata for ExtractResult."""
    if mrz is None:
        return "", {"mrz_valid": False, "mrz_raw": None}
    d = mrz.to_dict() if hasattr(mrz, "to_dict") else {}
    # Build a short text summary for .text (so LLM/entity flows get something)
    parts = []
    for key in ("number", "surname", "names", "nationality", "date_of_birth", "expiration_date", "sex"):
        v = d.get(key)
        if key == "number" and v is None:
            v = d.get("personal_number")
        if v is not None and str(v).strip():
            parts.append(f"{key}: {v}")
    text = "\n".join(parts) if parts else (getattr(mrz, "raw_text", None) or str(d) or "")
    metadata = {"mrz_valid": getattr(mrz, "valid", True), "mrz": d}
    if hasattr(mrz, "raw_text") and mrz.raw_text:
        metadata["mrz_raw"] = mrz.raw_text
    return text, metadata


class PassportEyeProcessor(BaseDocumentProcessor):
    """
    Extract MRZ (Machine Readable Zone) from passport/ID images using PassportEye.

    Uses OpenCV + pytesseract in MRZ-optimized mode. Best for the bottom 2–3 lines
    of the biodata page (MRZ). Returns ExtractResult with:
    - text: human-readable summary of MRZ fields (number, surname, names, DOB, expiry, etc.)
    - metadata.mrz: dict of parsed MRZ fields; metadata.mrz_valid: bool.

    Requires: PassportEye (pip install PassportEye or pip install gen-ai-framework[passporteye]),
    Tesseract binary (e.g. brew install tesseract). Supports image files only.
    """

    def extract(self, file_path: Union[str, Path]) -> ExtractResult:
        path = Path(file_path)
        if not path.exists():
            return ExtractResult.error_result(f"File not found: {path}")
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            return ExtractResult.error_result(
                f"Unsupported type: {path.suffix}. PassportEye supports images: {', '.join(IMAGE_EXTENSIONS)}."
            )
        try:
            from passporteye import read_mrz
        except ImportError:
            return ExtractResult.error_result(
                "PassportEye not installed. Install with: pip install PassportEye or pip install gen-ai-framework[passporteye]"
            )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                mrz = read_mrz(str(path))
            text, metadata = _mrz_to_text_and_metadata(mrz)
            return ExtractResult(text=text or "", metadata=metadata, error=None)
        except Exception as e:
            return ExtractResult.error_result(str(e))
