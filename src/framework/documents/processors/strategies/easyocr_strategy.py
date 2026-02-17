"""EasyOCR strategy: lazy-loaded reader, no image preprocessing."""

from typing import Any, List, Optional

import numpy as np

from ...types import OcrResult
from .base import OCRStrategy


def _parse_readtext_result(result: List[Any]) -> tuple[str, List[dict]]:
    """Parse EasyOCR readtext result (bbox, text, confidence) into text and details."""
    lines = [item[1] for item in result]
    text = "\n".join(lines)
    details = [{"text": item[1], "confidence": float(item[2])} for item in result]
    return text, details


class EasyOCRStrategy(OCRStrategy):
    """
    OCR backend using EasyOCR. Lazy-loads the reader on first use; no per-page
    re-initialization. Works on BGR images; no OpenCV preprocessing applied.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        gpu: bool = False,
    ) -> None:
        self._languages = languages or ["en"]
        self._gpu = gpu
        self._reader: Any = None

    def _get_reader(self):  # type: ignore[no-untyped-def]
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self._languages, gpu=self._gpu)
        return self._reader

    def extract(self, image: np.ndarray) -> OcrResult:
        """Run EasyOCR on BGR image. Converts grayscale to 3-channel if needed."""
        try:
            reader = self._get_reader()
            arr = np.asarray(image, dtype=np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            result = reader.readtext(arr)
            text, details = _parse_readtext_result(result)
            return OcrResult(text=text, details=details, error=None)
        except ImportError as e:
            return OcrResult(text="", error=f"EasyOCR not installed: {e}")
        except Exception as e:
            return OcrResult(text="", error=str(e))
