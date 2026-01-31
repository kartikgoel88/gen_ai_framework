"""Tests for framework evaluation (feedback store, golden datasets)."""

import json
from pathlib import Path

import pytest

from src.framework.evaluation.feedback_store import FeedbackEntry, FeedbackStore
from src.framework.evaluation.golden import (
    GoldenItem,
    GoldenRunResult,
    GoldenDatasetRunner,
    _exact_match,
    _keyword_match,
    _normalize,
)


def test_feedback_entry_to_dict_from_dict():
    """FeedbackEntry round-trip to_dict / from_dict."""
    e = FeedbackEntry(
        id="id1",
        session_id="s1",
        prompt="p",
        response="r",
        feedback="thumbs_up",
        score=1.0,
        metadata={"k": "v"},
        created_at="2024-01-01T00:00:00Z",
    )
    d = e.to_dict()
    assert d["id"] == "id1"
    assert d["prompt"] == "p"
    assert d["feedback"] == "thumbs_up"
    e2 = FeedbackEntry.from_dict(d)
    assert e2.id == e.id
    assert e2.prompt == e.prompt
    assert e2.feedback == e.feedback


def test_feedback_store_add_and_list(tmp_path):
    """FeedbackStore add appends entry, list_entries returns it."""
    path = tmp_path / "feedback.jsonl"
    store = FeedbackStore(path=str(path))
    eid = store.add(prompt="Q", response="A", feedback="thumbs_up")
    assert eid
    entries = store.list_entries(limit=10)
    assert len(entries) >= 1
    assert entries[0].prompt == "Q"
    assert entries[0].response == "A"
    assert entries[0].feedback == "thumbs_up"


def test_feedback_store_list_filter_session(tmp_path):
    """FeedbackStore list_entries filters by session_id."""
    path = tmp_path / "feedback.jsonl"
    store = FeedbackStore(path=str(path))
    store.add("Q1", "A1", session_id="s1")
    store.add("Q2", "A2", session_id="s2")
    entries = store.list_entries(limit=10, session_id="s1")
    assert all(e.session_id == "s1" for e in entries)
    assert len(entries) == 1


def test_normalize():
    """_normalize lowercases and collapses whitespace."""
    assert _normalize("  Hello   World  ") == "hello world"
    assert _normalize("") == ""


def test_exact_match():
    """_exact_match compares normalized strings or dicts."""
    assert _exact_match("Hello", "hello") is True
    assert _exact_match("a  b", "a b") is True
    assert _exact_match("x", "y") is False
    assert _exact_match({"a": 1}, {"a": 1}) is True


def test_keyword_match():
    """_keyword_match returns True if any keyword in normalized output."""
    assert _keyword_match("The answer is forty-two", ["forty-two"]) is True
    assert _keyword_match("No match here", ["forty"]) is False
    assert _keyword_match("anything", []) is True


def test_golden_run_result_pass_rate():
    """GoldenRunResult.pass_rate is passed/total."""
    r = GoldenRunResult(total=10, passed=8, failed=2, latency_seconds=1.0, compare_mode="keyword")
    assert r.pass_rate == 0.8
    r0 = GoldenRunResult(total=0, passed=0, failed=0, latency_seconds=0.0, compare_mode="exact")
    assert r0.pass_rate == 0.0


def test_golden_dataset_runner_run():
    """GoldenDatasetRunner run invokes run_fn and compares with expected_keywords."""
    items = [
        GoldenItem(id="1", inputs={"q": "x"}, expected_keywords=["yes"]),
        GoldenItem(id="2", inputs={"q": "y"}, expected_keywords=["no"]),
    ]

    def run_fn(inputs):
        if inputs.get("q") == "x":
            return "The answer is yes"
        return "The answer is no"

    runner = GoldenDatasetRunner(run_fn=run_fn, compare_mode="keyword")
    result = runner.run(items)
    assert result.total == 2
    assert result.passed == 2
    assert result.failed == 0
    assert result.compare_mode == "keyword"
    assert len(result.per_item) == 2


def test_golden_dataset_runner_load_dataset_json(tmp_path):
    """GoldenDatasetRunner.load_dataset loads JSON array."""
    path = tmp_path / "golden.json"
    path.write_text(json.dumps([
        {"id": "1", "inputs": {"q": "a"}, "expected_output": "b"},
        {"id": "2", "inputs": {"q": "c"}, "expected_keywords": ["d"]},
    ]))
    items = GoldenDatasetRunner.load_dataset(path)
    assert len(items) == 2
    assert items[0].id == "1"
    assert items[0].expected_output == "b"
    assert items[1].expected_keywords == ["d"]


def test_golden_dataset_runner_load_dataset_jsonl(tmp_path):
    """GoldenDatasetRunner.load_dataset loads JSONL."""
    path = tmp_path / "golden.jsonl"
    path.write_text('{"id":"1","inputs":{"q":"a"}}\n{"id":"2","inputs":{"q":"b"}}\n')
    items = GoldenDatasetRunner.load_dataset(path)
    assert len(items) == 2
    assert items[0].id == "1"
    assert items[1].id == "2"
