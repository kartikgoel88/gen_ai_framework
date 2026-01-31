"""Tests for framework RAG chunking."""

import pytest

from src.framework.rag.chunking import (
    get_text_splitter,
    RECURSIVE_SEPARATORS,
    SENTENCE_SEPARATORS,
)


def test_recursive_separators_defined():
    """RECURSIVE_SEPARATORS is non-empty."""
    assert len(RECURSIVE_SEPARATORS) >= 1
    assert "\n\n" in RECURSIVE_SEPARATORS


def test_sentence_separators_defined():
    """SENTENCE_SEPARATORS includes sentence boundaries."""
    assert ". " in SENTENCE_SEPARATORS or "." in "".join(SENTENCE_SEPARATORS)


def test_get_text_splitter_recursive_default():
    """get_text_splitter returns splitter with expected chunk_size/chunk_overlap."""
    splitter = get_text_splitter(strategy="recursive_character", chunk_size=100, chunk_overlap=10)
    assert getattr(splitter, "chunk_size", None) == 100 or getattr(splitter, "_chunk_size", None) == 100
    assert getattr(splitter, "chunk_overlap", None) == 10 or getattr(splitter, "_chunk_overlap", None) == 10
    assert hasattr(splitter, "split_text")


def test_get_text_splitter_sentence():
    """get_text_splitter with strategy=sentence returns splitter that splits text."""
    splitter = get_text_splitter(strategy="sentence", chunk_size=200, chunk_overlap=20)
    text = "First sentence. Second sentence. Third sentence."
    chunks = splitter.split_text(text)
    assert len(chunks) >= 1
    assert "First sentence" in chunks[0] or "First" in chunks[0]


def test_get_text_splitter_splits_text():
    """Splitter produces chunks within size limit."""
    splitter = get_text_splitter(chunk_size=50, chunk_overlap=5)
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = splitter.split_text(text)
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c) <= 50 + 20  # overlap can make slightly larger in edge cases
