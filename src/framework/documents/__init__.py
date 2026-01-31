"""Document processing: extract text and metadata from files."""

from .processor import DocumentProcessor
from .pdf_ocr_processor import PdfOcrProcessor
from .types import ExtractResult

__all__ = ["DocumentProcessor", "PdfOcrProcessor", "ExtractResult"]
