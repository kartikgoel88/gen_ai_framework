"""Tests for framework Docling processor."""

from pathlib import Path

import pytest

from src.framework.docling.processor import DoclingProcessor
from src.framework.docling.types import DoclingResult


def test_docling_processor_extract_file_not_found(tmp_path):
    """DoclingProcessor extract returns error when file does not exist."""
    proc = DoclingProcessor()
    path = tmp_path / "nonexistent.pdf"
    assert not path.exists()
    result = proc.extract(path)
    assert isinstance(result, DoclingResult)
    assert result.text == ""
    assert result.error is not None
    assert "not found" in result.error.lower() or str(path) in result.error


def test_docling_processor_extract_markdown_convenience(tmp_path):
    """extract_markdown calls extract with export_format=markdown."""
    proc = DoclingProcessor()
    path = tmp_path / "missing.pdf"
    result = proc.extract_markdown(path)
    assert result.text == ""
    assert result.error


def test_docling_processor_extract_text_convenience(tmp_path):
    """extract_text calls extract with export_format=text."""
    proc = DoclingProcessor()
    path = tmp_path / "missing.pdf"
    result = proc.extract_text(path)
    assert result.text == ""
    assert result.error


def test_docling_result_type():
    """DoclingResult has text and error attributes."""
    r = DoclingResult(text="hello", error=None)
    assert r.text == "hello"
    assert r.error is None
    r2 = DoclingResult(text="", error="failed")
    assert r2.error == "failed"
