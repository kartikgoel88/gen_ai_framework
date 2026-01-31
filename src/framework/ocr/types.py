"""OCR result type."""

from typing import Any, List, Optional


class OcrResult:
    """Result of OCR extraction."""

    def __init__(
        self,
        text: str,
        details: Optional[List[dict]] = None,
        error: Optional[str] = None,
    ):
        self.text = text
        self.details = details or []
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "details": self.details, "error": self.error}
