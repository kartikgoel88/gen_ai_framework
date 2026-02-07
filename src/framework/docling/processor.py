"""Docling processor: layout-aware document parsing with OCR support."""

from pathlib import Path
from typing import Union

from src.framework.documents.base import BaseDocumentProcessor
from src.framework.documents.types import ExtractResult

from .types import DoclingResult


class DoclingProcessor(BaseDocumentProcessor):
    """Parse documents with Docling (PDF, DOCX, etc.) and export to text/markdown."""

    def __init__(self, default_export_format: str = "markdown"):
        self._default_export_format = default_export_format

    def extract(self, file_path: Union[str, Path], export_format: str | None = None) -> ExtractResult:
        """
        Extract text from a document (base interface). Returns ExtractResult.
        Uses markdown export by default; set export_format to "text" for plain text.
        """
        fmt = export_format or self._default_export_format
        dr = self._extract_internal(file_path, export_format=fmt)
        return ExtractResult(
            text=dr.text,
            metadata=dr.metadata,
            error=dr.error,
        )

    def _extract_internal(
        self,
        file_path: Union[str, Path],
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

    def extract_markdown(self, file_path: Union[str, Path]) -> DoclingResult:
        """Convenience: extract as Markdown (returns DoclingResult)."""
        return self._extract_internal(file_path, export_format="markdown")

    def extract_text(self, file_path: Union[str, Path]) -> DoclingResult:
        """Convenience: extract as plain text (returns DoclingResult)."""
        return self._extract_internal(file_path, export_format="text")
