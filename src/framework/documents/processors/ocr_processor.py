"""
Unified OCR processor: PDFs (native text + OCR for scanned pages) and images.

Architecture:
- PDFProcessor: PDF extraction, native text, scanned-page detection.
- ImageProcessor: Load and optional enhancement (Real-ESRGAN + CLAHE).
- OCRStrategy (injected): EasyOCR, Tesseract, or PaddleOCR; no backend conditionals.
- Public API: extract(path) -> ExtractResult, extract_from_bytes(bytes) -> OcrResult.
"""

from pathlib import Path
from typing import List, Optional, Union

from ..base import BaseDocumentProcessor
from ..types import IMAGE_EXTENSIONS, ExtractResult, OcrResult, validate_file_path
from .factory import OcrBackend, create_ocr_strategy
from .image_processor import ImageProcessor
from .pdf_processor import PDFPageContent, PDFProcessor
from .strategies.base import OCRStrategy


class OcrProcessor(BaseDocumentProcessor):
    """
    Extract text from PDFs and images.

    - **PDFs**: Native text via PyMuPDF; pages detected as scanned (low text + image
      blocks) are rendered and run through the injected OCR strategy. No backend
      conditionals; strategy is created from ocr_backend (Dependency Injection).
    - **Images**: Loaded and optionally enhanced by ImageProcessor, then OCR via
      the same strategy. Lazy-loaded engines (EasyOCR, PaddleOCR); no repeated
      model init per page.

    Use extract(path) for both; extract_from_bytes(bytes) for in-memory images only.
    """

    def __init__(
        self,
        pdf_dpi: int = 300,
        pdf_min_text_len: int = 10,
        languages: Optional[List[str]] = None,
        gpu: bool = False,
        *,
        ocr_backend: OcrBackend = "easyocr",
        use_image_enhancement: bool = False,
        pdf_processor: Optional[PDFProcessor] = None,
        image_processor: Optional[ImageProcessor] = None,
        ocr_strategy: Optional[OCRStrategy] = None,
    ) -> None:
        self._languages = languages or ["en"]
        self._pdf_processor = pdf_processor or PDFProcessor(dpi=pdf_dpi, min_text_length=pdf_min_text_len)
        self._image_processor = image_processor or ImageProcessor(use_enhancement=use_image_enhancement)
        self._ocr_strategy = ocr_strategy or create_ocr_strategy(
            ocr_backend, languages=self._languages, gpu=gpu
        )

    def extract(self, file_path: Union[str, Path]) -> ExtractResult:
        """
        Extract text from a PDF or image file. Dispatches by extension.
        Returns ExtractResult with text, metadata, and optional error.
        """
        path = Path(file_path)
        allowed = (".pdf",) + IMAGE_EXTENSIONS
        err = validate_file_path(path, allowed)
        if err:
            return ExtractResult.error_result(err)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(path)
        if suffix in IMAGE_EXTENSIONS:
            return self._extract_ocr_image(path).to_extract_result()
        return ExtractResult.error_result(
            f"Unsupported type: {suffix}. Use .pdf or image (e.g. .png, .jpg)."
        )

    def _extract_pdf(self, path: Path) -> ExtractResult:
        """Extract from PDF: native text or OCR on scanned pages via injected strategy."""
        pages_content, error = self._pdf_processor.extract_pages(path)
        if error:
            return ExtractResult.error_result(error)
        if not pages_content:
            return ExtractResult("", metadata={"pages": 0, "ocr_pages": []}, error=None)

        full_text_parts: List[str] = []
        ocr_pages: List[int] = []
        num_pages = len(pages_content)

        for content in pages_content:
            if content.native_text is not None:
                full_text_parts.append(content.native_text)
                continue
            if content.image is None:
                full_text_parts.append("")
                continue
            img = self._image_processor.enhance(content.image)
            ocr_result = self._ocr_strategy.extract_for_pdf_page(img)
            text_ocr = ocr_result.text if not ocr_result.error else ""
            full_text_parts.append(text_ocr or "")
            ocr_pages.append(content.page_num)

        full_text = "\n".join(full_text_parts).strip()
        return ExtractResult(
            full_text,
            metadata={"pages": num_pages, "ocr_pages": ocr_pages},
            error=None,
        )

    def _extract_ocr_image(self, image_path: Union[str, Path]) -> OcrResult:
        """Load image (with optional enhancement), then run injected OCR strategy."""
        path = Path(image_path)
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            return OcrResult(text="", error=f"Unsupported image type: {path.suffix}")
        img, load_err = self._image_processor.load(path)
        if load_err:
            return OcrResult(text="", error=load_err)
        if img is None:
            return OcrResult(text="", error="Could not load image")
        return self._ocr_strategy.extract(img)

    def extract_from_bytes(self, image_bytes: bytes) -> OcrResult:
        """Run OCR on image bytes. Load and optional enhancement via ImageProcessor, then strategy."""
        img, load_err = self._image_processor.load_bytes(image_bytes)
        if load_err:
            return OcrResult(text="", error=load_err)
        if img is None:
            return OcrResult(text="", error="Could not decode image bytes")
        return self._ocr_strategy.extract(img)
