"""
Image enhancement for OCR: Real-ESRGAN (super-resolution) + CLAHE (contrast).

Step 1: Optional Real-ESRGAN upscaling (requires realesrgan + basicsr).
Step 2: CLAHE (Contrast Limited Adaptive Histogram Equalization) via OpenCV.

Use enhance_image() to run the full pipeline; each step is optional and
gracefully skipped if dependencies are missing.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


def _apply_clahe(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE to the L channel in LAB space; keeps color, improves contrast for text."""
    import cv2

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_channel)
    lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def _apply_realesrgan(img_bgr: np.ndarray, scale: int = 4, tile: int = 0) -> Optional[np.ndarray]:
    """
    Upscale image with Real-ESRGAN. Returns enhanced BGR numpy array or None if unavailable.
    Requires: realesrgan, basicsr, torch.
    """
    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        return None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        model_path = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x{scale}plus.pth"
        upsampler = RealESRGANer(scale=scale, model_path=model_path, model=model, tile=tile, tile_pad=10, device=device)
        output, _ = upsampler.enhance(img_bgr, outscale=scale)
        return output
    except Exception:
        return None


def enhance_image(
    image_input: Union[np.ndarray, str, Path],
    use_realesrgan: bool = True,
    use_clahe: bool = True,
    realesrgan_scale: int = 4,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Run image enhancement pipeline: Real-ESRGAN (optional) then CLAHE (optional).

    Args:
        image_input: BGR numpy array (H, W, 3), or path to image file.
        use_realesrgan: If True, run Real-ESRGAN upscaling first (skipped if deps missing).
        use_clahe: If True, apply CLAHE to improve contrast for text.
        realesrgan_scale: Scale factor for Real-ESRGAN (2 or 4).
        clahe_clip_limit: CLAHE clip limit (higher = more contrast).
        clahe_tile_grid_size: CLAHE tile grid size.

    Returns:
        BGR numpy array (uint8). Same as input if both steps are disabled or unavailable.
    """
    import cv2

    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise ValueError(f"Could not load image: {image_input}")
    else:
        img = np.asarray(image_input, dtype=np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim != 3:
            raise ValueError("Image must be 2D (grayscale) or 3D (BGR) array")

    if use_realesrgan:
        upscaled = _apply_realesrgan(img, scale=realesrgan_scale)
        if upscaled is not None:
            img = upscaled

    if use_clahe:
        img = _apply_clahe(img, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid_size)

    return img
