"""PaddleOCR strategy: lazy-loaded engine, no binarization preprocessing."""

from typing import Any, List, Optional

import numpy as np

from ...types import OcrResult
from .base import OCRStrategy


class PaddleOCRStrategy(OCRStrategy):
    """
    OCR backend using PaddleOCR. Lazy-loads the engine on first use; no
    per-page re-initialization. Does not use adaptiveThreshold; works on
    raw BGR images. Optional angle classification (cls=True).
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        gpu: bool = False,
    ) -> None:
        self._languages = languages or ["en"]
        self._gpu = gpu
        self._paddle_ocr: Any = None

    def _get_engine(self):  # type: ignore[no-untyped-def]
        if self._paddle_ocr is None:
            from paddleocr import PaddleOCR
            lang = "en" if "en" in self._languages else self._languages[0]
            self._paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=self._gpu,
                show_log=False,
            )
        return self._paddle_ocr

    def extract(self, image: np.ndarray) -> OcrResult:
        """Run PaddleOCR on BGR image. No preprocessing."""
        try:
            ocr = self._get_engine()
            result = ocr.ocr(image, cls=True)
            if not result or not result[0]:
                return OcrResult(text="", details=[], error=None)
            lines: List[str] = []
            details: List[dict] = []
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]
                    conf = float(line[1][1]) if len(line[1]) > 1 else 0.0
                    lines.append(text)
                    details.append({"text": text, "confidence": conf})
            return OcrResult(text="\n".join(lines), details=details, error=None)
        except ImportError as e:
            return OcrResult(text="", error=f"PaddleOCR not installed: {e}")
        except Exception as e:
            return OcrResult(text="", error=str(e))
