"""Configurable text chunking strategies (LangChain-compatible)."""

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Strategy: recursive_character (default) vs sentence (sentence-boundary first)
RECURSIVE_SEPARATORS = ["\n\n", "\n", " ", ""]
SENTENCE_SEPARATORS = ["\n\n", ". ", "! ", "? ", "\n", " ", ""]


def get_text_splitter(
    strategy: str = "recursive_character",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs: Any,
) -> RecursiveCharacterTextSplitter:
    """Return a LangChain text splitter for the given strategy."""
    strategy = (strategy or "recursive_character").lower().strip()
    separators = SENTENCE_SEPARATORS if strategy == "sentence" else RECURSIVE_SEPARATORS
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        **kwargs,
    )
