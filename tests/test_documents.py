"""Tests for framework document processing."""

import tempfile
from pathlib import Path

import pytest

from src.framework.documents.types import ExtractResult
from src.framework.documents.processor import DocumentProcessor


def test_extract_result_to_dict():
    """ExtractResult to_dict returns text, metadata, error."""
    r = ExtractResult("hello", metadata={"k": "v"}, error=None)
    d = r.to_dict()
    assert d["text"] == "hello"
    assert d["metadata"] == {"k": "v"}
    assert d["error"] is None

    r2 = ExtractResult("", metadata={}, error="failed")
    d2 = r2.to_dict()
    assert d2["error"] == "failed"


def test_document_processor_extract_txt(tmp_path):
    """DocumentProcessor extract .txt returns content."""
    f = tmp_path / "doc.txt"
    f.write_text("Hello world\nLine 2", encoding="utf-8")
    proc = DocumentProcessor(upload_dir=str(tmp_path))
    result = proc.extract(f)
    assert result.text == "Hello world\nLine 2"
    assert result.error is None
    assert "encoding" in result.metadata or result.metadata


def test_document_processor_unsupported_type(tmp_path):
    """DocumentProcessor extract unsupported suffix returns error."""
    f = tmp_path / "file.xyz"
    f.write_bytes(b"x")
    proc = DocumentProcessor(upload_dir=str(tmp_path))
    result = proc.extract(f)
    assert result.text == ""
    assert result.error is not None
    assert "Unsupported" in result.error or "xyz" in result.error


def test_document_processor_save_upload(tmp_path):
    """DocumentProcessor save_upload writes file and returns path."""
    proc = DocumentProcessor(upload_dir=str(tmp_path))
    path = proc.save_upload(b"content", "test.txt")
    assert path.exists()
    assert path.read_bytes() == b"content"
    assert path.name == "test.txt"


def test_document_processor_save_upload_deduplication(tmp_path):
    """save_upload deduplicates by appending _1, _2 when file exists."""
    proc = DocumentProcessor(upload_dir=str(tmp_path))
    p1 = proc.save_upload(b"a", "dup.txt")
    p2 = proc.save_upload(b"b", "dup.txt")
    assert p1 != p2
    assert p1.read_bytes() == b"a"
    assert p2.read_bytes() == b"b"
    assert "dup_1" in str(p2) or "dup_" in p2.stem
