"""Shared JSON extraction from text (e.g. LLM or OCR output)."""

import json
import re
from typing import Any


def parse_json_from_text(text: str) -> dict[str, Any] | None:
    """
    Extract the first JSON object from text. Handles markdown code fences and surrounding text.

    Args:
        text: Raw text that may contain a JSON object.

    Returns:
        Parsed dict, or None if no valid JSON object found.
    """
    if not text or not text.strip():
        return None
    text = text.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    m = re.search(r"(\{[\s\S]*\})", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


def parse_json_from_response(text: str, default_raw: bool = True) -> dict[str, Any]:
    """
    Extract a single JSON object from model response text. Used by LLM invoke_structured.

    Args:
        text: Raw response text (may contain markdown or extra text).
        default_raw: If True and no JSON found, return {"raw": text}. If False, return {}.

    Returns:
        Parsed dict, or {"raw": text} / {} when no JSON found.
    """
    text = (text or "").strip()
    parsed = parse_json_from_text(text)
    if parsed is not None:
        return parsed
    return {"raw": text} if default_raw else {}
