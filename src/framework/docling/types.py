"""Docling result type."""

from typing import Any, Optional


class DoclingResult:
    """Result of Docling extraction."""

    def __init__(
        self,
        text: str,
        metadata: Optional[dict] = None,
        error: Optional[str] = None,
    ):
        self.text = text
        self.metadata = metadata or {}
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "metadata": self.metadata, "error": self.error}
