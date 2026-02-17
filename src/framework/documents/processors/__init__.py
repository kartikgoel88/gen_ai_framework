"""Document processors: PDF, image loading/enhancement, OCR strategies and factory."""

from .factory import OcrBackend, OCRFactory, create_ocr_strategy
from .image_processor import ImageProcessor
from .ocr_processor import OcrProcessor
from .pdf_processor import PDFPageContent, PDFProcessor
from .strategies import EasyOCRStrategy, OCRStrategy, PaddleOCRStrategy, TesseractOCRStrategy

__all__ = [
    "OcrBackend",
    "OCRFactory",
    "create_ocr_strategy",
    "ImageProcessor",
    "OcrProcessor",
    "PDFPageContent",
    "PDFProcessor",
    "OCRStrategy",
    "EasyOCRStrategy",
    "TesseractOCRStrategy",
    "PaddleOCRStrategy",
]
