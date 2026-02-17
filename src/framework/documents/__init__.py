"""Document processing: extract text and metadata from files (PDF + images via OCR, MRZ via PassportEye)."""

from .base import BaseDocumentProcessor
from .image_enhancement import enhance_image
from .ocr_processor import OcrProcessor
from .passport_eye_processor import PassportEyeProcessor
from .processor import DocumentProcessor
from .types import IMAGE_EXTENSIONS, ExtractResult, OcrResult

__all__ = [
    "BaseDocumentProcessor",
    "DocumentProcessor",
    "ExtractResult",
    "IMAGE_EXTENSIONS",
    "OcrProcessor",
    "OcrResult",
    "PassportEyeProcessor",
    "enhance_image",
]
