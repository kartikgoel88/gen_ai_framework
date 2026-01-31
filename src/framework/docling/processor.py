"""Docling processor: layout-aware document parsing with OCR support."""

from pathlib import Path
from typing import Optional

from .types import DoclingResult


class DoclingProcessor:
    """Parse documents with Docling (PDF, DOCX, etc.) and export to text/markdown."""

    def __init__(self):
        pass

    def extract(
        self,
        file_path: str | Path,
        export_format: str = "markdown",
    ) -> DoclingResult:
        """Parse document with Docling. export_format: 'markdown' | 'text'."""
        path = Path(file_path)
        if not path.exists():
            return DoclingResult(text="", error=f"File not found: {path}")
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            conv_result = converter.convert(str(path))
            doc = conv_result.document
            if export_format == "markdown":
                text = doc.export_to_markdown()
            else:
                text = doc.export_to_text()
            return DoclingResult(
                text=text,
                metadata={"pages": len(getattr(doc, "pages", {})) or None},
            )
        except Exception as e:
            return DoclingResult(text="", error=str(e))

    def extract_markdown(self, file_path: str | Path) -> DoclingResult:
        """Convenience: extract as Markdown."""
        return self.extract(file_path, export_format="markdown")

    def extract_text(self, file_path: str | Path) -> DoclingResult:
        """Convenience: extract as plain text."""
        return self.extract(file_path, export_format="text")
