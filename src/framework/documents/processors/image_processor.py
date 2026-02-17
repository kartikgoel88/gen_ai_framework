"""Image loading and enhancement: single place for OpenCV decode + optional Real-ESRGAN/CLAHE."""

import io
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


class ImageProcessor:
    """
    Handles image loading (path or bytes) and optional enhancement pipeline.
    Centralizes enhancement and OpenCV preprocessing so OCR strategies do not
    duplicate this logic. Enhancement is configurable and isolated from OCR.
    """

    def __init__(
        self,
        use_enhancement: bool = False,
        *,
        use_realesrgan: bool = True,
        use_clahe: bool = True,
        realesrgan_scale: int = 4,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> None:
        self._use_enhancement = use_enhancement
        self._use_realesrgan = use_realesrgan
        self._use_clahe = use_clahe
        self._realesrgan_scale = realesrgan_scale
        self._clahe_clip_limit = clahe_clip_limit
        self._clahe_tile_grid_size = clahe_tile_grid_size

    def load(self, image_path: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Load image from path, optionally run enhancement. Returns (image, error).
        Image is BGR numpy array (H, W, 3). On failure, image is None and error is set.
        """
        import cv2
        path = Path(image_path)
        img = cv2.imread(str(path))
        if img is None:
            return None, "Could not load image"
        if self._use_enhancement:
            img, enh_err = self._enhance(img)
            if enh_err is not None:
                pass  # keep original image on enhancement failure
        return img, None

    def enhance(self, img: np.ndarray) -> np.ndarray:
        """
        Run enhancement pipeline on an in-memory BGR image (e.g. PDF page).
        No-op if use_enhancement is False. Returns enhanced image or original on failure.
        """
        if not self._use_enhancement:
            return img
        out, _ = self._enhance(img)
        return out

    def load_bytes(self, image_bytes: bytes) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Load image from in-memory bytes (e.g. PNG/JPEG), optionally enhance.
        Returns (BGR numpy array, error). Decodes via PIL then converts to BGR.
        """
        import cv2
        try:
            from PIL import Image
        except ImportError:
            return None, "PIL (Pillow) required for loading image bytes"
        try:
            pil = Image.open(io.BytesIO(image_bytes))
            arr = np.array(pil)
            if arr.ndim == 2:
                img = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            else:
                img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception as e:
            return None, str(e)
        if self._use_enhancement:
            img, enh_err = self._enhance(img)
            if enh_err is not None:
                pass
        return img, None

    def _enhance(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """
        Run enhancement pipeline (Real-ESRGAN + CLAHE). Returns (enhanced_image, error).
        On dependency failure or exception, returns (original_image, error_message).
        """
        try:
            from ...image_enhancement import enhance_image
            out = enhance_image(
                img,
                use_realesrgan=self._use_realesrgan,
                use_clahe=self._use_clahe,
                realesrgan_scale=self._realesrgan_scale,
                clahe_clip_limit=self._clahe_clip_limit,
                clahe_tile_grid_size=self._clahe_tile_grid_size,
            )
            return out, None
        except ImportError as e:
            return img, f"Enhancement dependencies missing: {e}"
        except Exception as e:
            return img, str(e)
