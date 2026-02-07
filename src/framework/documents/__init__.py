"""Document processing: extract text and metadata from files (PDF + images via OCR)."""

from .base import BaseDocumentProcessor
from .ocr_processor import OcrProcessor
from .processor import DocumentProcessor
from .types import IMAGE_EXTENSIONS, ExtractResult, OcrResult

__all__ = [
    "BaseDocumentProcessor",
    "DocumentProcessor",
    "ExtractResult",
    "IMAGE_EXTENSIONS",
    "OcrProcessor",
    "OcrResult",
]
