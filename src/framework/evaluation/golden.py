"""Golden datasets: run batch/agents/RAG on fixed inputs and compare outputs (regression)."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..llm.base import LLMClient
from ..rag.base import RAGClient


@dataclass
class GoldenItem:
    """Single golden example: inputs and optional expected output for regression."""

    id: str
    inputs: dict[str, Any]
    expected_output: Optional[Any] = None  # str or dict (for structured)
    expected_keywords: Optional[list[str]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GoldenRunResult:
    """Result of running a golden dataset: pass/fail counts and per-item details."""

    total: int
    passed: int
    failed: int
    latency_seconds: float
    compare_mode: str  # exact | keyword | diff
    per_item: list[dict[str, Any]] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def _normalize(s: str) -> str:
    return " ".join(re.split(r"\s+", (s or "").lower().strip()))


def _exact_match(pred: Any, expected: Any) -> bool:
    if isinstance(pred, dict) and isinstance(expected, dict):
        return json.dumps(pred, sort_keys=True) == json.dumps(expected, sort_keys=True)
    return _normalize(str(pred)) == _normalize(str(expected))


def _keyword_match(pred: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    p = _normalize(str(pred))
    return any(_normalize(k) in p for k in keywords)


class GoldenDatasetRunner:
    """Run a target (rag | batch | agent) on fixed inputs and compare outputs for regression."""

    def __init__(
        self,
        run_fn: Callable[[dict[str, Any]], Any],
        compare_mode: str = "keyword",
    ):
        """
        Args:
            run_fn: Function that takes inputs dict and returns output (str or dict).
            compare_mode: exact | keyword | diff. exact = strict match; keyword = expected_keywords in output; diff = store diff for review.
        """
        self._run_fn = run_fn
        self._compare_mode = (compare_mode or "keyword").lower().strip()

    def run(
        self,
        items: list[GoldenItem],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> GoldenRunResult:
        import time
        start = time.perf_counter()
        total = len(items)
        passed = 0
        failed = 0
        per_item: list[dict[str, Any]] = []

        for i, item in enumerate(items):
            try:
                output = self._run_fn(item.inputs)
            except Exception as e:
                output = None
                per_item.append({
                    "id": item.id,
                    "passed": False,
                    "error": str(e),
                    "expected": item.expected_output,
                    "expected_keywords": item.expected_keywords,
                    "output": None,
                })
                failed += 1
                if progress_callback:
                    progress_callback(i + 1, total)
                continue

            ok = False
            if self._compare_mode == "exact" and item.expected_output is not None:
                ok = _exact_match(output, item.expected_output)
            elif self._compare_mode == "keyword" and item.expected_keywords:
                ok = _keyword_match(str(output), item.expected_keywords)
            elif self._compare_mode == "keyword" and item.expected_output is not None:
                ok = _exact_match(output, item.expected_output) or _keyword_match(
                    str(output), [str(item.expected_output)]
                )
            elif item.expected_output is None and not item.expected_keywords:
                ok = True
            else:
                ok = _exact_match(output, item.expected_output)

            if ok:
                passed += 1
            else:
                failed += 1

            per_item.append({
                "id": item.id,
                "passed": ok,
                "output": output,
                "expected": item.expected_output,
                "expected_keywords": item.expected_keywords,
            })
            if progress_callback:
                progress_callback(i + 1, total)

        elapsed = time.perf_counter() - start
        return GoldenRunResult(
            total=total,
            passed=passed,
            failed=failed,
            latency_seconds=elapsed,
            compare_mode=self._compare_mode,
            per_item=per_item,
        )

    @staticmethod
    def load_dataset(path: str | Path) -> list[GoldenItem]:
        """Load golden dataset from JSON array or JSONL. Each row: id, inputs, expected_output?, expected_keywords?."""
        path = Path(path)
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8").strip()
        items: list[GoldenItem] = []
        if text.startswith("["):
            data = json.loads(text)
            for row in data:
                items.append(GoldenItem(
                    id=row.get("id", str(len(items))),
                    inputs=row.get("inputs", {}),
                    expected_output=row.get("expected_output"),
                    expected_keywords=row.get("expected_keywords"),
                    metadata=row.get("metadata", {}),
                ))
        else:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                items.append(GoldenItem(
                    id=row.get("id", str(len(items))),
                    inputs=row.get("inputs", {}),
                    expected_output=row.get("expected_output"),
                    expected_keywords=row.get("expected_keywords"),
                    metadata=row.get("metadata", {}),
                ))
        return items
