"""OCR processor using EasyOCR (images) and optional PDF page images."""

from pathlib import Path
from typing import List, Optional

from .types import OcrResult


class OcrProcessor:
    """Extract text from images using EasyOCR."""

    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False):
        self._languages = languages or ["en"]
        self._gpu = gpu
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self._languages, gpu=self._gpu)
        return self._reader

    def extract(self, image_path: str | Path) -> OcrResult:
        """Run OCR on an image file. Returns OcrResult with text and optional word boxes."""
        path = Path(image_path)
        if not path.exists():
            return OcrResult(text="", error=f"File not found: {path}")
        suffix = path.suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"):
            return OcrResult(text="", error=f"Unsupported image type: {suffix}")
        try:
            reader = self._get_reader()
            result = reader.readtext(str(path))
            # result: list of (bbox, text, confidence)
            lines = [item[1] for item in result]
            text = "\n".join(lines)
            details = [{"text": item[1], "confidence": float(item[2])} for item in result]
            return OcrResult(text=text, details=details)
        except Exception as e:
            return OcrResult(text="", error=str(e))

    def extract_from_bytes(self, image_bytes: bytes) -> OcrResult:
        """Run OCR on image bytes (e.g. in-memory)."""
        try:
            import numpy as np
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(image_bytes))
            arr = np.array(img)
            reader = self._get_reader()
            result = reader.readtext(arr)
            lines = [item[1] for item in result]
            text = "\n".join(lines)
            details = [{"text": item[1], "confidence": float(item[2])} for item in result]
            return OcrResult(text=text, details=details)
        except Exception as e:
            return OcrResult(text="", error=str(e))
