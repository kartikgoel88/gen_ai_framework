"""PDF extraction and native text detection; identifies scanned pages for OCR."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np


@dataclass
class PDFPageContent:
    """Content for a single PDF page: either native text or a rendered image for OCR."""
    page_num: int  # 1-based
    native_text: Optional[str]  # set when page is not scanned
    image: Optional[np.ndarray]  # BGR image when page is scanned; None otherwise


class PDFProcessor:
    """
    Handles only PDF extraction and native text detection. Renders pages to images
    when they are detected as scanned (low native text + image blocks). Does not
    run OCR; callers use OCR strategy on page.image when present.
    """

    def __init__(
        self,
        dpi: int = 300,
        min_text_length: int = 10,
    ) -> None:
        self._dpi = dpi
        self._min_text_length = min_text_length

    def _is_scanned_page(self, page: Any, native_text: str) -> bool:
        """
        Return True if the page should be treated as scanned (run OCR).
        Conditions: native text length below threshold AND page contains image blocks.
        """
        text_stripped = (native_text or "").strip()
        if len(text_stripped) >= self._min_text_length:
            return False
        # PyMuPDF: get_images() returns list of (xref, ...) for images on the page
        try:
            images = page.get_images()
        except Exception:
            images = []
        return len(images) > 0

    def extract_pages(self, path: Path) -> Tuple[List[PDFPageContent], Optional[str]]:
        """
        Open PDF and extract per-page content: native text or rendered image for scanned pages.
        Returns (list of PDFPageContent, error). On success error is None.
        """
        try:
            import fitz
        except ImportError:
            return [], "PyMuPDF (fitz) not installed. Install with: pip install pymupdf"

        try:
            import cv2
        except ImportError as e:
            return [], f"opencv required for PDF rendering: {e}"

        if path.suffix.lower() != ".pdf":
            return [], f"Not a PDF: {path.suffix}"

        try:
            doc = fitz.open(path)
        except Exception as e:
            return [], str(e)

        result: List[PDFPageContent] = []
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                native_text = page.get_text("text") or ""
                one_based = page_num + 1

                if not self._is_scanned_page(page, native_text):
                    result.append(PDFPageContent(page_num=one_based, native_text=native_text, image=None))
                    continue

                pix = page.get_pixmap(dpi=self._dpi)
                img = np.frombuffer(pix.tobytes(), dtype=np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                if img is None:
                    result.append(PDFPageContent(page_num=one_based, native_text="", image=None))
                else:
                    result.append(PDFPageContent(page_num=one_based, native_text=None, image=img))
            doc.close()
            return result, None
        except Exception as e:
            try:
                doc.close()
            except Exception:
                pass
            return [], str(e)
