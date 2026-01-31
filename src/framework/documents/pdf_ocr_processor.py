"""PDF processor using PyMuPDF (fitz) with pytesseract OCR fallback for scanned pages."""

from pathlib import Path
from typing import Optional

from .types import ExtractResult


class PdfOcrProcessor:
    """
    Extract text from PDFs: native text via PyMuPDF first, then pytesseract OCR
    fallback for pages with no or insufficient text (e.g. scanned receipts).

    Requires: PyMuPDF (pymupdf), pytesseract, opencv-python-headless, numpy.
    System: Tesseract binary must be installed (e.g. apt install tesseract-ocr, brew install tesseract).
    """

    def __init__(self, dpi: int = 300, min_text_len: int = 10):
        """
        Args:
            dpi: Resolution for rendering pages to image when using OCR.
            min_text_len: If native text length per page is below this, use OCR for that page.
        """
        self._dpi = dpi
        self._min_text_len = min_text_len

    def extract(self, file_path: str | Path) -> ExtractResult:
        """
        Extract text from a PDF. For each page: use PyMuPDF native text if present,
        otherwise render page to image and run pytesseract OCR (with OpenCV preprocessing).
        Returns ExtractResult with text and metadata (e.g. pages, ocr_pages).
        """
        path = Path(file_path)
        if not path.exists():
            return ExtractResult("", metadata={}, error=f"File not found: {path}")
        if path.suffix.lower() != ".pdf":
            return ExtractResult("", metadata={}, error=f"Not a PDF: {path.suffix}")

        try:
            import fitz
        except ImportError:
            return ExtractResult(
                "",
                metadata={},
                error="PyMuPDF (fitz) not installed. Install with: pip install pymupdf",
            )
        try:
            import pytesseract
            import cv2
            import numpy as np
        except ImportError as e:
            return ExtractResult(
                "",
                metadata={},
                error=f"pytesseract/opencv/numpy required for OCR fallback: {e}",
            )

        full_text = ""
        ocr_pages: list[int] = []
        try:
            doc = fitz.open(path)
            num_pages = len(doc)
            for page_num in range(num_pages):
                page = doc[page_num]
                native_text = page.get_text("text") or ""
                if native_text.strip() and len(native_text.strip()) >= self._min_text_len:
                    full_text += native_text + "\n"
                    continue
                # OCR fallback for this page
                pix = page.get_pixmap(dpi=self._dpi)
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
            meta = {"pages": num_pages, "ocr_pages": ocr_pages}
            return ExtractResult(full_text.strip(), metadata=meta, error=None)
        except Exception as e:
            return ExtractResult("", metadata={}, error=str(e))
