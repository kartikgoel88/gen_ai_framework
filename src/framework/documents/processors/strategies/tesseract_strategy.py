"""Tesseract OCR strategy: preprocessing (denoise, adaptive threshold) for document/passport text."""

from pathlib import Path
from typing import List

import numpy as np

from ...types import OcrResult
from .base import OCRStrategy

# Tesseract config: PSM 3 = full auto, OEM 3 = LSTM+legacy
_TESS_CONFIG = "--psm 3 --oem 3"
_TESS_LANG = "eng"
_MIN_SIZE_FOR_UPSCALE = 1600


class TesseractOCRStrategy(OCRStrategy):
    """
    OCR backend using pytesseract. Applies OpenCV preprocessing: upscale small
    images, denoise, grayscale, and optional adaptive threshold. Uses two
    passes (grayscale + threshold) and merges results. Expects preprocessing
    when used for PDF page OCR (binarized image).
    """

    def __init__(self) -> None:
        pass

    @property
    def needs_preprocessing_for_pdf(self) -> bool:
        return True

    def _preprocess_for_image(self, img: np.ndarray) -> np.ndarray:
        """Upscale small images, denoise, grayscale. Returns grayscale image."""
        import cv2
        h, w = img.shape[:2]
        if max(h, w) < _MIN_SIZE_FOR_UPSCALE:
            scale = _MIN_SIZE_FOR_UPSCALE / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        except cv2.error:
            pass
        return gray

    def _preprocess_for_pdf(self, img: np.ndarray) -> np.ndarray:
        """Binarize for PDF page: grayscale + adaptive threshold. Used only for PDF OCR."""
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 2,
        )

    def _merge_pass_results(self, texts: List[str]) -> str:
        """Dedupe lines and prefer longer result; join distinct lines."""
        seen: set[str] = set()
        merged_lines: List[str] = []
        for block in texts:
            for line in block.splitlines():
                line = line.strip()
                if not line or line in seen:
                    continue
                seen.add(line)
                merged_lines.append(line)
        return "\n".join(merged_lines) if merged_lines else (texts[0] if texts else "")

    def extract(self, image: np.ndarray) -> OcrResult:
        """Run Tesseract with two-pass preprocessing (grayscale + adaptive thresh) and merge."""
        import pytesseract
        try:
            gray = self._preprocess_for_image(image)
            texts: List[str] = []
            t1 = (pytesseract.image_to_string(gray, lang=_TESS_LANG, config=_TESS_CONFIG) or "").strip()
            if t1:
                texts.append(t1)
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31, 2,
            )
            t2 = (pytesseract.image_to_string(thresh, lang=_TESS_LANG, config=_TESS_CONFIG) or "").strip()
            if t2 and t2 != t1:
                texts.append(t2)
            text = self._merge_pass_results(texts)
            return OcrResult(text=text.strip(), details=[], error=None)
        except ImportError as e:
            return OcrResult(text="", error=f"pytesseract/opencv required: {e}")
        except Exception as e:
            return OcrResult(text="", error=str(e))

    def extract_for_pdf_page(self, image: np.ndarray) -> OcrResult:
        """
        Run Tesseract on a PDF-rendered page with binarization only (no denoise/upscale).
        Called by OcrProcessor when strategy.needs_preprocessing_for_pdf is True.
        """
        import pytesseract
        try:
            gray = self._preprocess_for_pdf(image)
            text = (pytesseract.image_to_string(gray, lang=_TESS_LANG, config=_TESS_CONFIG) or "").strip()
            return OcrResult(text=text, details=[], error=None)
        except ImportError as e:
            return OcrResult(text="", error=f"pytesseract/opencv required: {e}")
        except Exception as e:
            return OcrResult(text="", error=str(e))

    def extract_from_path(self, image_path: Path) -> OcrResult:
        """Load image, run full image preprocessing (upscale/denoise), then OCR."""
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            return OcrResult(text="", error="Could not load image")
        return self.extract(img)
