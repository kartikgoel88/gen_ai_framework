"""Factory for OCR strategies: returns the correct strategy based on config (Dependency Injection)."""

from typing import List, Literal, Optional

from .strategies.base import OCRStrategy
from .strategies.easyocr_strategy import EasyOCRStrategy
from .strategies.paddle_strategy import PaddleOCRStrategy
from .strategies.tesseract_strategy import TesseractOCRStrategy

OcrBackend = Literal["easyocr", "pytesseract", "paddle"]


def create_ocr_strategy(
    backend: OcrBackend,
    *,
    languages: Optional[List[str]] = None,
    gpu: bool = False,
) -> OCRStrategy:
    """
    Create an OCR strategy for the given backend. Use this to inject the strategy
    into OcrProcessor (no backend conditionals in business logic).

    Args:
        backend: One of "easyocr", "pytesseract", "paddle".
        languages: Language list for EasyOCR/PaddleOCR (e.g. ["en"]).
        gpu: Whether to use GPU for EasyOCR/PaddleOCR.

    Returns:
        OCRStrategy implementation. Same instance can be reused (lazy-loaded engines).
    """
    if backend == "easyocr":
        return EasyOCRStrategy(languages=languages, gpu=gpu)
    if backend == "pytesseract":
        return TesseractOCRStrategy()
    if backend == "paddle":
        return PaddleOCRStrategy(languages=languages, gpu=gpu)
    raise ValueError(f"Unknown OCR backend: {backend}. Use one of: easyocr, pytesseract, paddle")


class OCRFactory:
    """Factory for OCR strategies. Use create_ocr_strategy() or OCRFactory.get_strategy()."""

    @staticmethod
    def get_strategy(
        backend: OcrBackend,
        *,
        languages: Optional[List[str]] = None,
        gpu: bool = False,
    ) -> OCRStrategy:
        """Return the OCR strategy for the given backend (lazy-loaded engines)."""
        return create_ocr_strategy(backend, languages=languages, gpu=gpu)
