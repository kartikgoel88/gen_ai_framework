"""Docling processor: layout-aware document parsing with OCR support."""

import warnings
from pathlib import Path
from typing import Union

# Suppress strict_text deprecation from docling/OCR stack (may be emitted at import or during convert)
warnings.filterwarnings("ignore", message=".*strict_text.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*strict_text.*")

from src.framework.documents.base import BaseDocumentProcessor
from src.framework.documents.types import ExtractResult, validate_file_path

from .types import DoclingResult


def _make_converter_with_ocr():
    """Build DocumentConverter with OCR enabled for PDF and images (so images yield text)."""
    from docling.document_converter import DocumentConverter

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TesseractCliOcrOptions,
        )
        from docling.document_converter import PdfFormatOption

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
        format_options = {InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        # Enable OCR for images if ImageFormatOption exists and accepts pipeline_options
        try:
            from docling.document_converter import ImageFormatOption
            format_options[InputFormat.IMAGE] = ImageFormatOption(pipeline_options=pipeline_options)
        except (ImportError, AttributeError, TypeError):
            pass
        return DocumentConverter(format_options=format_options)
    except Exception:
        return DocumentConverter()


class DoclingProcessor(BaseDocumentProcessor):
    """Parse documents with Docling (PDF, DOCX, images). Layout-aware, runs locally — PII-safe (no cloud/API)."""

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
        err = validate_file_path(path, allowed_extensions=None)
        if err:
            return DoclingResult(text="", error=err)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*strict_text.*", category=DeprecationWarning)
                warnings.filterwarnings("ignore", message=".*strict_text.*")
                warnings.filterwarnings("ignore", category=DeprecationWarning, module="docling.*")
                converter = _make_converter_with_ocr()
                conv_result = converter.convert(str(path))
            doc = conv_result.document
            if export_format == "markdown":
                text = doc.export_to_markdown()
            else:
                text = doc.export_to_text()
            return DoclingResult(
                text=text or "",
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
