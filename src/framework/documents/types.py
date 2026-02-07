"""Types and shared constants for document processing."""

from pathlib import Path
from typing import Any, List, Optional, Tuple

# File extensions supported by OcrProcessor (images).
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")


def validate_file_path(
    path: Path,
    allowed_extensions: Optional[Tuple[str, ...]] = None,
) -> Optional[str]:
    """
    Validate file path exists and (optionally) has an allowed extension.

    Args:
        path: Path to validate.
        allowed_extensions: If given, path.suffix must be in this tuple (lowercased). None = only check existence.

    Returns:
        None if valid; otherwise an error message string.
    """
    if not path.exists():
        return f"File not found: {path}"
    if allowed_extensions is not None and path.suffix.lower() not in allowed_extensions:
        return f"Unsupported type: {path.suffix}. Allowed: {', '.join(allowed_extensions)}."
    return None


class ExtractResult:
    """Result of document text extraction."""

    def __init__(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ):
        self.text = text
        self.metadata = metadata or {}
        self.error = error

    @classmethod
    def error_result(cls, message: str, metadata: Optional[dict[str, Any]] = None) -> "ExtractResult":
        """Build an ExtractResult representing a failure (no text, error set)."""
        return cls("", metadata=metadata or {}, error=message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "error": self.error,
        }


class OcrResult:
    """Result of OCR extraction (images). Used by OcrProcessor for extract_from_bytes and internal use."""

    def __init__(
        self,
        text: str,
        details: Optional[List[dict]] = None,
        error: Optional[str] = None,
    ):
        self.text = text
        self.details = details or []
        self.error = error

    def to_extract_result(self) -> ExtractResult:
        """Convert to canonical ExtractResult (details in metadata)."""
        return ExtractResult(
            text=self.text,
            metadata={"details": self.details} if self.details else {},
            error=self.error,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "details": self.details, "error": self.error}
