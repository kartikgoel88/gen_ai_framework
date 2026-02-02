"""Convert Confluence storage HTML to plain text for RAG."""

from __future__ import annotations

import re
from html import unescape


def html_to_text(html: str) -> str:
    """
    Convert Confluence storage (HTML) body to plain text.
    Uses BeautifulSoup if available for robust stripping; otherwise regex fallback.
    """
    if not (html or "").strip():
        return ""

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        # Remove script/style
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    except ImportError:
        text = _strip_html_regex(html)

    text = unescape(text)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_html_regex(html: str) -> str:
    """Fallback: strip tags with regex when BeautifulSoup is not available."""
    # Remove script/style blocks
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Replace block elements with newline
    html = re.sub(r"</(p|div|br|tr|li|h[1-6])[^>]*>", "\n", html, flags=re.IGNORECASE)
    # Remove all tags
    html = re.sub(r"<[^>]+>", " ", html)
    return html
