#!/usr/bin/env python3
"""
Production-ready OCR script using DeepSeek-OCR in two modes:
  - Local: Hugging Face model (transformers + torch), runs fully offline.
  - Endpoint: OpenAI-compatible API (e.g. Ollama), sends base64 image.

Usage:
  Local mode (default):
    python scripts/deepseek_ocr.py --image_path /path/to/image.png
    python scripts/deepseek_ocr.py --image_path /path/to/image.png --mode local --device cuda
    python scripts/deepseek_ocr.py --batch /path/to/folder --output_file results.json

  Endpoint mode:
    python scripts/deepseek_ocr.py --image_path /path/to/image.png --mode endpoint
    python scripts/deepseek_ocr.py --image_path /path/to/image.png --mode endpoint --base_url http://localhost:11434/v1 --model_name deepseek-ocr
    python scripts/deepseek_ocr.py --batch /path/to/folder --mode endpoint --base_url https://api.openai.com/v1 --api_key sk-...
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal, Optional

import requests

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def _setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LOCAL_MODEL = "deepseek-ai/DeepSeek-OCR"
DEFAULT_ENDPOINT_MODEL = "deepseek-ocr"
DEFAULT_OCR_PROMPT = "Extract all readable text from this image."
DEFAULT_MAX_NEW_TOKENS = 512

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
DeviceKind = Literal["cpu", "cuda", "auto"]
ModeKind = Literal["local", "endpoint"]


# -----------------------------------------------------------------------------
# DeepSeekOCR (local model)
# -----------------------------------------------------------------------------

class DeepSeekOCR:
    """
    Local DeepSeek-OCR inference using Hugging Face transformers + torch.
    Loads model once; supports single image or folder. Uses half precision and
    autocast on GPU when available.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_MODEL,
        device: DeviceKind = "auto",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        base_size: int = 1024,
        image_size: int = 1024,
        crop_mode: bool = False,
        prompt_template: str = "<image>\n<|grounding|>Convert the document to markdown.",
    ) -> None:
        self.model_name = model_name
        self.device_kind = device
        self.max_new_tokens = max_new_tokens
        self.base_size = base_size
        self.image_size = image_size
        self.crop_mode = crop_mode
        self.prompt_template = prompt_template
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None

    def _resolve_device(self) -> Any:
        import torch
        if self.device_kind == "cpu":
            return torch.device("cpu")
        if self.device_kind == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
            return torch.device("cuda")
        # auto
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load_model(self) -> None:
        """Load tokenizer and model once. Uses half precision on GPU and optional autocast."""
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer

        # DeepSeek-OCR model code imports LlamaFlashAttention2, removed in transformers 4.48+
        tx_version = getattr(transformers, "__version__", "0")
        if tx_version >= "4.48":
            raise ImportError(
                f"DeepSeek-OCR requires transformers 4.46.x–4.47.x (you have {tx_version}). "
                "Install the extra: uv sync --extra deepseek-ocr"
            )

        if self._model is not None:
            logger.debug("Model already loaded, skipping")
            return

        self._device = self._resolve_device()
        logger.info("Loading tokenizer: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        logger.info("Loading model: %s (device=%s)", self.model_name, self._device)
        use_flash = False
        if self._device.type == "cuda":
            try:
                import flash_attn  # noqa: F401
                use_flash = True
            except ImportError:
                logger.warning("flash_attn not installed; using sdpa. Install for faster GPU inference.")

        load_kw: dict[str, Any] = {
            "trust_remote_code": True,
            "use_safetensors": True,
        }
        if use_flash:
            load_kw["_attn_implementation"] = "flash_attention_2"

        try:
            self._model = AutoModel.from_pretrained(self.model_name, **load_kw)
        except ImportError as e:
            if "LlamaFlashAttention2" in str(e) or "flash_attention" in str(e).lower():
                logger.warning(
                    "Flash attention not compatible with this transformers version; loading with default attention. "
                    "For flash_attn use: torch 2.6.0, transformers 4.47.1, flash-attn 2.7.3"
                )
                load_kw.pop("_attn_implementation", None)
                self._model = AutoModel.from_pretrained(self.model_name, **load_kw)
            else:
                raise
        self._model = self._model.eval().to(self._device)

        if self._device.type == "cuda":
            self._model = self._model.to(torch.bfloat16)
        elif self._device.type == "mps":
            try:
                self._model = self._model.to(torch.float16)
            except Exception:
                pass
        logger.info("Model loaded successfully")

    def extract_text(self, image: "PIL.Image.Image") -> str:
        """Extract text from a PIL Image. Loads model on first call."""
        self.load_model()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f.name, format="PNG")
            path = Path(f.name)
        try:
            return self.extract_from_path(path)
        finally:
            path.unlink(missing_ok=True)

    def extract_from_path(self, path: Path) -> str:
        """Extract text from an image file. Loads model on first call."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if path.suffix.lower() not in IMAGE_EXTENSIONS and path.suffix.lower() != ".pdf":
            logger.warning("Unusual image extension: %s", path.suffix)

        self.load_model()
        import torch
        prompt = self.prompt_template
        output_dir = tempfile.mkdtemp()
        try:
            with torch.no_grad():
                if self._device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        res = self._model.infer(
                            self._tokenizer,
                            prompt=prompt,
                            image_file=str(path),
                            output_path=output_dir,
                            base_size=self.base_size,
                            image_size=self.image_size,
                            crop_mode=self.crop_mode,
                            save_results=False,
                            test_compress=False,
                        )
                else:
                    res = self._model.infer(
                        self._tokenizer,
                        prompt=prompt,
                        image_file=str(path),
                        output_path=output_dir,
                        base_size=self.base_size,
                        image_size=self.image_size,
                        crop_mode=self.crop_mode,
                        save_results=False,
                        test_compress=False,
                    )
        finally:
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
        if res is None:
            return ""
        return (res if isinstance(res, str) else str(res)).strip()

    def extract_folder(self, folder: Path) -> dict[str, str]:
        """Process all images in folder. Returns mapping filename -> extracted text."""
        folder = Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder}")
        self.load_model()
        results: dict[str, str] = {}
        files = sorted(
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        for i, path in enumerate(files):
            logger.info("Processing %s (%d/%d)", path.name, i + 1, len(files))
            try:
                text = self.extract_from_path(path)
                results[path.name] = text
            except Exception as e:
                logger.exception("Failed to process %s: %s", path.name, e)
                results[path.name] = ""
        return results


# -----------------------------------------------------------------------------
# EndpointOCRClient
# -----------------------------------------------------------------------------

class EndpointOCRClient:
    """
    OCR via an OpenAI-compatible chat/completions endpoint. Encodes image to
    base64 and sends a single user message with text + image_url.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_ENDPOINT_MODEL,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    def encode_image_to_base64(self, path: Path) -> str:
        """Read image file and return base64-encoded string (no data URL prefix)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        raw = path.read_bytes()
        return base64.standard_b64encode(raw).decode("ascii")

    def _mime_type(self, path: Path) -> str:
        suf = path.suffix.lower()
        if suf in (".png",):
            return "image/png"
        if suf in (".jpg", ".jpeg",):
            return "image/jpeg"
        if suf in (".webp",):
            return "image/webp"
        if suf in (".gif",):
            return "image/gif"
        return "image/png"

    def send_request(self, base64_image: str, mime: str = "image/png") -> str:
        """
        Send OpenAI-style chat request with one user message containing
        text + image_url (data URL). Returns extracted text from first choice.
        """
        url = f"{self.base_url}/chat/completions"
        data_url = f"data:{mime};base64,{base64_image}"
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": DEFAULT_OCR_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
            "max_tokens": self.max_new_tokens,
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices")
        if not choices:
            raise ValueError("Endpoint returned no choices")
        content = choices[0].get("message", {}).get("content")
        if content is None:
            return ""
        return content.strip() if isinstance(content, str) else str(content).strip()

    def extract_from_path(self, path: Path) -> str:
        """Encode image and send to endpoint; return extracted text."""
        path = Path(path)
        b64 = self.encode_image_to_base64(path)
        mime = self._mime_type(path)
        return self.send_request(b64, mime=mime)

    def extract_folder(self, folder: Path) -> dict[str, str]:
        """Process all images in folder. Returns mapping filename -> extracted text."""
        folder = Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder}")
        results: dict[str, str] = {}
        files = sorted(
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        for i, path in enumerate(files):
            logger.info("Processing %s (%d/%d)", path.name, i + 1, len(files))
            try:
                text = self.extract_from_path(path)
                results[path.name] = text
            except Exception as e:
                logger.exception("Failed to process %s: %s", path.name, e)
                results[path.name] = ""
        return results


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DeepSeek-OCR: extract text from images (local model or OpenAI-compatible endpoint).",
    )
    p.add_argument("--image_path", type=Path, help="Path to a single image file")
    p.add_argument(
        "--mode",
        choices=["local", "endpoint"],
        default="local",
        help="local = Hugging Face model offline; endpoint = OpenAI-compatible API",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device for local mode (auto = prefer GPU)",
    )
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max tokens to generate")
    p.add_argument("--output_file", type=Path, help="Write extracted text (or JSON for --batch) here")
    p.add_argument("--batch", type=Path, help="Process all images in this folder")
    # Endpoint-only
    p.add_argument("--base_url", default=DEFAULT_BASE_URL, help="Endpoint base URL (e.g. Ollama)")
    p.add_argument("--api_key", default=None, help="Optional API key for endpoint")
    p.add_argument(
        "--model_name",
        default=None,
        help="Local default: deepseek-ai/DeepSeek-OCR; Endpoint default: deepseek-ocr",
    )
    return p.parse_args()


def main() -> int:
    _setup_logging()
    args = parse_args()

    if args.batch is not None and args.image_path is not None:
        logger.error("Use either --image_path or --batch, not both")
        return 1
    if args.batch is None and args.image_path is None:
        logger.error("Provide --image_path or --batch")
        return 1

    model_name = args.model_name
    if model_name is None:
        model_name = DEFAULT_LOCAL_MODEL if args.mode == "local" else DEFAULT_ENDPOINT_MODEL

    if args.mode == "local":
        ocr = DeepSeekOCR(
            model_name=model_name,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        if args.batch is not None:
            if not args.batch.is_dir():
                logger.error("--batch must be a directory: %s", args.batch)
                return 1
            results = ocr.extract_folder(args.batch)
            out = json.dumps(results, indent=2, ensure_ascii=False)
            print(out)
            if args.output_file:
                args.output_file.write_text(out, encoding="utf-8")
                logger.info("Wrote %s", args.output_file)
        else:
            path = Path(args.image_path)
            if not path.exists():
                logger.error("Image not found: %s", path)
                return 1
            text = ocr.extract_from_path(path)
            print(text)
            if args.output_file:
                args.output_file.write_text(text, encoding="utf-8")
                logger.info("Wrote %s", args.output_file)
    else:
        client = EndpointOCRClient(
            base_url=args.base_url,
            api_key=args.api_key,
            model_name=model_name,
            max_new_tokens=args.max_new_tokens,
        )
        if args.batch is not None:
            if not args.batch.is_dir():
                logger.error("--batch must be a directory: %s", args.batch)
                return 1
            results = client.extract_folder(args.batch)
            out = json.dumps(results, indent=2, ensure_ascii=False)
            print(out)
            if args.output_file:
                args.output_file.write_text(out, encoding="utf-8")
                logger.info("Wrote %s", args.output_file)
        else:
            path = Path(args.image_path)
            if not path.exists():
                logger.error("Image not found: %s", path)
                return 1
            try:
                text = client.extract_from_path(path)
            except requests.RequestException as e:
                logger.error("Endpoint request failed: %s", e)
                return 1
            print(text)
            if args.output_file:
                args.output_file.write_text(text, encoding="utf-8")
                logger.info("Wrote %s", args.output_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())


# -----------------------------------------------------------------------------
# requirements.txt content (for this script)
# -----------------------------------------------------------------------------
#
# transformers>=4.46.0
# torch>=2.0.0
# Pillow>=10.0.0
# requests>=2.28.0
#
# Optional for faster GPU inference (CUDA):
#   flash-attn --no-build-isolation
#
# -----------------------------------------------------------------------------
# How to run
# -----------------------------------------------------------------------------
#
# Local mode: Downloads and runs the DeepSeek-OCR model on your machine (GPU if
# available). No network after model load. Requires: transformers, torch, Pillow.
#   python scripts/deepseek_ocr.py --image_path /path/to/image.png
#
# Endpoint mode: Sends the image (base64) to an OpenAI-compatible API (e.g. Ollama
# serving a vision model). No local model load. Requires: requests, Pillow.
#   python scripts/deepseek_ocr.py --image_path /path/to/image.png --mode endpoint
#
# -----------------------------------------------------------------------------
# Example CLI usage
# -----------------------------------------------------------------------------
#
# LOCAL MODE (loads Hugging Face model, runs offline):
#
#   # Single image, print text
#   python scripts/deepseek_ocr.py --image_path /path/to/receipt.png
#
#   # Single image, save to file
#   python scripts/deepseek_ocr.py --image_path receipt.png --output_file out.txt
#
#   # Force CPU
#   python scripts/deepseek_ocr.py --image_path receipt.png --device cpu
#
#   # GPU with more tokens
#   python scripts/deepseek_ocr.py --image_path receipt.png --device cuda --max_new_tokens 1024
#
#   # Process folder, save JSON (filename -> text)
#   python scripts/deepseek_ocr.py --batch ./images --output_file results.json
#
# ENDPOINT MODE (sends request to OpenAI-compatible API, e.g. Ollama):
#
#   # Default Ollama base URL
#   python scripts/deepseek_ocr.py --image_path receipt.png --mode endpoint
#
#   # Custom base URL and model
#   python scripts/deepseek_ocr.py --image_path receipt.png --mode endpoint \
#     --base_url http://localhost:11434/v1 --model_name deepseek-ocr
#
#   # With API key (e.g. OpenAI-compatible proxy)
#   python scripts/deepseek_ocr.py --image_path receipt.png --mode endpoint \
#     --base_url https://api.openai.com/v1 --api_key sk-... --model_name gpt-4o
#
#   # Batch folder via endpoint
#   python scripts/deepseek_ocr.py --batch ./images --mode endpoint --output_file results.json
#
