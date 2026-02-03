"""Evaluation harness: run RAG/LLM on a dataset and compute metrics."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..llm.base import LLMClient
from ..rag.base import RAGClient


# Type for (model_id, LLMClient) pairs used in multi-model eval
ModelSpec = tuple[str, LLMClient]


@dataclass
class EvalDatasetItem:
    """Single eval example: question and expected reference(s)."""

    question: str
    expected_answer: Optional[str] = None  # exact or reference answer
    expected_keywords: Optional[list[str]] = None  # at least one should appear
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Aggregate eval result for a run."""

    total: int
    exact_match: int
    keyword_match: int  # predicted contains at least one expected keyword
    latency_seconds: float
    per_item: list[dict[str, Any]] = field(default_factory=list)

    @property
    def exact_match_rate(self) -> float:
        return self.exact_match / self.total if self.total else 0.0

    @property
    def keyword_match_rate(self) -> float:
        return self.keyword_match / self.total if self.total else 0.0


def _normalize(s: str) -> str:
    """Normalize for comparison: lowercase, collapse whitespace."""
    return " ".join(re.split(r"\s+", (s or "").lower().strip()))


def _exact_match(pred: str, expected: str) -> bool:
    return _normalize(pred) == _normalize(expected)


def _keyword_match(pred: str, keywords: list[str]) -> bool:
    if not keywords:
        return False
    pred_n = _normalize(pred)
    return any(_normalize(k) in pred_n for k in keywords)


class EvalHarness:
    """Run RAG/LLM on a list of (question, expected) and compute metrics."""

    def __init__(
        self,
        llm: LLMClient,
        rag: Optional[RAGClient] = None,
        prompt_template: Optional[str] = None,
        top_k: int = 4,
    ):
        self._llm = llm
        self._rag = rag
        self._top_k = top_k
        self._prompt_template = prompt_template or (
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

    def _answer(self, question: str) -> str:
        if self._rag:
            chunks = self._rag.retrieve(question, top_k=self._top_k)
            context = "\n\n".join(c.get("content", "") for c in chunks)
            prompt = self._prompt_template.format(context=context, question=question)
        else:
            prompt = question
        return self._llm.invoke(prompt).strip()

    def run(
        self,
        items: list[EvalDatasetItem],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> EvalResult:
        import time

        start = time.perf_counter()
        total = len(items)
        exact_match = 0
        keyword_match = 0
        per_item: list[dict[str, Any]] = []

        for i, item in enumerate(items):
            pred = self._answer(item.question)
            em = False
            if item.expected_answer is not None:
                em = _exact_match(pred, item.expected_answer)
                if em:
                    exact_match += 1
            km = False
            if item.expected_keywords:
                km = _keyword_match(pred, item.expected_keywords)
                if km:
                    keyword_match += 1
            elif item.expected_answer is None and not item.expected_keywords:
                keyword_match += 1  # no reference: count as keyword match for rate
            per_item.append(
                {
                    "question": item.question,
                    "expected_answer": item.expected_answer,
                    "expected_keywords": item.expected_keywords,
                    "predicted": pred,
                    "exact_match": em,
                    "keyword_match": km,
                }
            )
            if progress_callback:
                progress_callback(i + 1, total)

        elapsed = time.perf_counter() - start
        return EvalResult(
            total=total,
            exact_match=exact_match,
            keyword_match=keyword_match,
            latency_seconds=elapsed,
            per_item=per_item,
        )

    @staticmethod
    def load_dataset(path: str | Path) -> list[EvalDatasetItem]:
        """Load dataset from JSON (array) or JSONL. Each line/object: question, optional expected_answer, optional expected_keywords."""
        path = Path(path)
        if not path.exists():
            return []
        items: list[EvalDatasetItem] = []
        text = path.read_text(encoding="utf-8").strip()
        if text.startswith("["):
            data = json.loads(text)
            for row in data:
                items.append(
                    EvalDatasetItem(
                        question=row.get("question", ""),
                        expected_answer=row.get("expected_answer"),
                        expected_keywords=row.get("expected_keywords"),
                        metadata=row.get("metadata", {}),
                    )
                )
        else:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                items.append(
                    EvalDatasetItem(
                        question=row.get("question", ""),
                        expected_answer=row.get("expected_answer"),
                        expected_keywords=row.get("expected_keywords"),
                        metadata=row.get("metadata", {}),
                    )
                )
        return items


def evaluate_multiple_models(
    items: list[EvalDatasetItem],
    models: list[ModelSpec],
    *,
    rag: Optional[RAGClient] = None,
    prompt_template: Optional[str] = None,
    top_k: int = 4,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> dict[str, EvalResult]:
    """Run the eval harness on the same dataset for multiple models.

    Args:
        items: Eval dataset (question, expected_answer or expected_keywords).
        models: List of (model_id, LLMClient) to compare, e.g. [
            ("gpt-4", llm_gpt4),
            ("gpt-3.5", llm_gpt35),
        ].
        rag: Optional RAG client (same for all models).
        prompt_template: Optional prompt template for RAG.
        top_k: Retrieval top_k when using RAG.
        progress_callback: Optional callback (model_id, current, total) per model.

    Returns:
        Dict mapping model_id -> EvalResult for each model.
    """
    results: dict[str, EvalResult] = {}
    for model_id, llm in models:
        harness = EvalHarness(
            llm=llm,
            rag=rag,
            prompt_template=prompt_template,
            top_k=top_k,
        )
        cb: Optional[Callable[[int, int], None]] = None
        if progress_callback:
            mid = model_id  # bind for closure
            def _cb(current: int, total: int) -> None:
                progress_callback(mid, current, total)
            cb = _cb
        results[model_id] = harness.run(items, progress_callback=cb)
    return results
