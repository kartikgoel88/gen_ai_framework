"""Types for document processing."""

from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "error": self.error,
        }
