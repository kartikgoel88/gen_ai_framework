"""Unified OCR processor: PDFs (PyMuPDF + pytesseract fallback) and images (EasyOCR or pytesseract)."""

import io
from pathlib import Path
from typing import Any, List, Optional, Union

from .base import BaseDocumentProcessor
from .types import IMAGE_EXTENSIONS, ExtractResult, OcrResult


def _parse_readtext_result(result: List[Any]) -> tuple[str, List[dict]]:
    """Parse EasyOCR readtext result (bbox, text, confidence) into text and details."""
    lines = [item[1] for item in result]
    text = "\n".join(lines)
    details = [{"text": item[1], "confidence": float(item[2])} for item in result]
    return text, details


class OcrProcessor(BaseDocumentProcessor):
    """
    Extract text from PDFs and images.

    - **PDFs**: PyMuPDF native text first, then pytesseract + OpenCV OCR fallback
      for pages with little or no text (e.g. scanned). Requires: pymupdf, pytesseract,
      opencv-python-headless, numpy; Tesseract binary installed.
    - **Images**: EasyOCR by default, or pytesseract when use_pytesseract_for_images=True.
      Pytesseract for images requires: pytesseract, opencv-python-headless, numpy;
      Tesseract binary installed (e.g. brew install tesseract).

    Use extract(path) for both; extract_from_bytes(bytes) for in-memory images only.
    """

    def __init__(
        self,
        pdf_dpi: int = 300,
        pdf_min_text_len: int = 10,
        languages: Optional[List[str]] = None,
        gpu: bool = False,
        use_pytesseract_for_images: bool = False,
    ):
        self._pdf_dpi = pdf_dpi
        self._pdf_min_text_len = pdf_min_text_len
        self._languages = languages or ["en"]
        self._gpu = gpu
        self._use_pytesseract_for_images = use_pytesseract_for_images
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self._languages, gpu=self._gpu)
        return self._reader

    def extract(self, file_path: Union[str, Path]) -> ExtractResult:
        """
        Extract text from a PDF or image file. Dispatches by extension.
        Returns ExtractResult with text, metadata, and optional error.
        """
        path = Path(file_path)
        if not path.exists():
            return ExtractResult.error_result(f"File not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(path)
        if suffix in IMAGE_EXTENSIONS:
            if self._use_pytesseract_for_images:
                return self._extract_ocr_pytesseract(path).to_extract_result()
            return self._extract_ocr(path).to_extract_result()
        return ExtractResult.error_result(
            f"Unsupported type: {suffix}. Use .pdf or image (e.g. .png, .jpg)."
        )

    def _extract_pdf(self, path: Path) -> ExtractResult:
        """PDF: PyMuPDF native text + pytesseract OCR fallback per page."""
        if path.suffix.lower() != ".pdf":
            return ExtractResult.error_result(f"Not a PDF: {path.suffix}")
        try:
            import fitz
        except ImportError:
            return ExtractResult.error_result(
                "PyMuPDF (fitz) not installed. Install with: pip install pymupdf"
            )
        try:
            import cv2
            import numpy as np
            import pytesseract
        except ImportError as e:
            return ExtractResult.error_result(
                f"pytesseract/opencv/numpy required for PDF OCR fallback: {e}"
            )
        full_text = ""
        ocr_pages: list[int] = []
        try:
            doc = fitz.open(path)
            num_pages = len(doc)
            for page_num in range(num_pages):
                page = doc[page_num]
                native_text = page.get_text("text") or ""
                if native_text.strip() and len(native_text.strip()) >= self._pdf_min_text_len:
                    full_text += native_text + "\n"
                    continue
                pix = page.get_pixmap(dpi=self._pdf_dpi)
                img = np.frombuffer(pix.tobytes(), dtype=np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                if img is None:
                    full_text += "\n"
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.adaptiveThreshold(
                    gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    31, 2,
                )
                text_ocr = pytesseract.image_to_string(gray, lang="eng")
                full_text += (text_ocr or "") + "\n"
                ocr_pages.append(page_num + 1)
            doc.close()
            return ExtractResult(
                full_text.strip(),
                metadata={"pages": num_pages, "ocr_pages": ocr_pages},
                error=None,
            )
        except Exception as e:
            return ExtractResult.error_result(str(e))

    def _extract_ocr_pytesseract(self, image_path: Union[str, Path]) -> OcrResult:
        """Image: pytesseract + OpenCV with preprocessing for better passport/document text."""
        path = Path(image_path)
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            return OcrResult(text="", error=f"Unsupported image type: {path.suffix}")
        try:
            import cv2
            import pytesseract
            img = cv2.imread(str(path))
            if img is None:
                return OcrResult(text="", error="Could not load image")
            # Upscale small images so small passport text is readable by Tesseract
            h, w = img.shape[:2]
            scale = 1.0
            if max(h, w) < 1600:
                scale = 1600 / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Denoise to reduce photo noise (helps passports)
            try:
                gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
            except Exception:
                pass
            # Tesseract config: PSM 3 = full auto, OEM 3 = LSTM+legacy
            config = "--psm 3 --oem 3"
            texts: List[str] = []
            # Pass 1: grayscale (often best for photos)
            t1 = (pytesseract.image_to_string(gray, lang="eng", config=config) or "").strip()
            if t1:
                texts.append(t1)
            # Pass 2: adaptive threshold (helps low-contrast or scanned look)
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31, 2,
            )
            t2 = (pytesseract.image_to_string(thresh, lang="eng", config=config) or "").strip()
            if t2 and t2 != t1:
                texts.append(t2)
            # Merge: prefer longer result; if similar length, join and dedupe lines
            seen: set[str] = set()
            merged_lines: List[str] = []
            for block in texts:
                for line in block.splitlines():
                    line = line.strip()
                    if not line or line in seen:
                        continue
                    seen.add(line)
                    merged_lines.append(line)
            text = "\n".join(merged_lines) if merged_lines else (texts[0] if texts else "")
            return OcrResult(text=text.strip(), details=[], error=None)
        except ImportError as e:
            return OcrResult(text="", error=f"pytesseract/opencv required for image OCR: {e}")
        except Exception as e:
            return OcrResult(text="", error=str(e))

    def _extract_ocr(self, image_path: Union[str, Path]) -> OcrResult:
        """Image: EasyOCR. Returns OcrResult (use .to_extract_result() for ExtractResult)."""
        path = Path(image_path)
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            return OcrResult(text="", error=f"Unsupported image type: {path.suffix}")
        try:
            reader = self._get_reader()
            result = reader.readtext(str(path))
            text, details = _parse_readtext_result(result)
            return OcrResult(text=text, details=details)
        except Exception as e:
            return OcrResult(text="", error=str(e))

    def extract_from_bytes(self, image_bytes: bytes) -> OcrResult:
        """Run OCR on image bytes (e.g. in-memory). Returns OcrResult."""
        try:
            import numpy as np
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes))
            arr = np.array(img)
            reader = self._get_reader()
            result = reader.readtext(arr)
            text, details = _parse_readtext_result(result)
            return OcrResult(text=text, details=details)
        except Exception as e:
            return OcrResult(text="", error=str(e))
