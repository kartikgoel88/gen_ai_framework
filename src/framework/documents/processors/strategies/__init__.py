"""OCR strategies: EasyOCR, Tesseract, PaddleOCR."""

from .base import OCRStrategy
from .easyocr_strategy import EasyOCRStrategy
from .paddle_strategy import PaddleOCRStrategy
from .tesseract_strategy import TesseractOCRStrategy

__all__ = [
    "OCRStrategy",
    "EasyOCRStrategy",
    "TesseractOCRStrategy",
    "PaddleOCRStrategy",
]
