"""OCR strategy interface: abstract base for EasyOCR, Tesseract, and PaddleOCR backends."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ...types import OcrResult


class OCRStrategy(ABC):
    """
    Abstract base class for OCR backends.

    Implementations run text detection on a BGR numpy image (H, W, 3).
    Preprocessing (e.g. enhancement, grayscale) is handled by callers or by
    the strategy itself when backend-specific (e.g. Tesseract preprocessing).
    """

    @abstractmethod
    def extract(self, image: np.ndarray) -> OcrResult:
        """
        Run OCR on a single BGR image (numpy array, uint8).

        Args:
            image: BGR image as numpy array (H, W, 3).

        Returns:
            OcrResult with text, details, and optional error.
        """
        ...

    def extract_from_path(self, image_path: Path) -> OcrResult:
        """
        Optional: load image from path and run OCR. Default implementation
        loads with OpenCV and calls extract(). Override for path-based APIs.

        Args:
            image_path: Path to image file.

        Returns:
            OcrResult with text, details, and optional error.
        """
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            return OcrResult(text="", error="Could not load image")
        return self.extract(img)

    def extract_for_pdf_page(self, image: np.ndarray) -> OcrResult:
        """
        Run OCR on a PDF-rendered page. Override in strategies that need
        different preprocessing for PDF (e.g. binarization only). Default: extract(image).
        """
        return self.extract(image)

    @property
    def needs_preprocessing_for_pdf(self) -> bool:
        """
        Whether this strategy expects preprocessed (e.g. binarized) images
        when used for PDF page OCR. Tesseract benefits from it; Paddle/EasyOCR do not.
        """
        return False
