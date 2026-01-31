"""A/B testing: run two prompts (or versions) and compare metrics."""

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..llm.base import LLMClient
from ..observability.eval import EvalHarness, EvalDatasetItem, EvalResult


@dataclass
class ABTestResult:
    """Result of A/B test: metrics for variant A and B, winner, delta."""

    variant_a: str
    variant_b: str
    metric: str
    score_a: float
    score_b: float
    winner: str  # "a" | "b" | "tie"
    delta: float
    details_a: Optional[dict] = None
    details_b: Optional[dict] = None


class ABTestRunner:
    """Run two prompt variants on the same inputs and compare a metric (exact_match, keyword_match, latency)."""

    def __init__(
        self,
        llm: LLMClient,
        metric: str = "keyword_match",
        rag=None,
        eval_top_k: int = 4,
    ):
        self._llm = llm
        self._metric = (metric or "keyword_match").lower().strip()
        self._rag = rag
        self._eval_top_k = eval_top_k

    def _run_prompt(self, prompt_template: str, items: list[EvalDatasetItem]) -> tuple[list[str], float]:
        """Run prompt (format with item.question) for each item; return (responses, total_latency)."""
        import time
        start = time.perf_counter()
        responses = []
        for item in items:
            prompt = prompt_template.format(question=item.question)
            if self._rag:
                chunks = self._rag.retrieve(item.question, top_k=self._eval_top_k)
                context = "\n\n".join(c.get("content", "") for c in chunks)
                prompt = f"Context:\n{context}\n\n{prompt}"
            responses.append(self._llm.invoke(prompt).strip())
        elapsed = time.perf_counter() - start
        return responses, elapsed

    def _scores_from_result(self, result: EvalResult) -> tuple[float, float]:
        """Return (primary_metric_score, latency_seconds)."""
        if self._metric == "exact_match":
            return result.exact_match_rate, result.latency_seconds
        if self._metric == "latency":
            return -result.latency_seconds, result.latency_seconds  # higher is better -> negate
        return result.keyword_match_rate, result.latency_seconds

    def run(
        self,
        prompt_a: str,
        prompt_b: str,
        items: list[EvalDatasetItem],
        variant_a_name: str = "A",
        variant_b_name: str = "B",
    ) -> ABTestResult:
        """Run both prompts on items; compare by configured metric. Prompt template must contain {question}."""
        responses_a, latency_a = self._run_prompt(prompt_a, items)
        responses_b, latency_b = self._run_prompt(prompt_b, items)
        # Build EvalResult-like aggregates for A and B
        exact_a = sum(1 for i, item in enumerate(items) if item.expected_answer and self._normalize(responses_a[i]) == self._normalize(item.expected_answer))
        exact_b = sum(1 for i, item in enumerate(items) if item.expected_answer and self._normalize(responses_b[i]) == self._normalize(item.expected_answer))
        def kw_match(resp, item):
            if not item.expected_keywords:
                return True
            rn = self._normalize(resp)
            return any(self._normalize(k) in rn for k in item.expected_keywords)
        kw_a = sum(1 for i, item in enumerate(items) if kw_match(responses_a[i], item))
        kw_b = sum(1 for i, item in enumerate(items) if kw_match(responses_b[i], item))
        n = len(items) or 1
        score_a = (exact_a / n) if self._metric == "exact_match" else (-latency_a if self._metric == "latency" else (kw_a / n))
        score_b = (exact_b / n) if self._metric == "exact_match" else (-latency_b if self._metric == "latency" else (kw_b / n))
        delta = score_a - score_b
        if delta > 0:
            winner = "a"
        elif delta < 0:
            winner = "b"
        else:
            winner = "tie"
        return ABTestResult(
            variant_a=variant_a_name,
            variant_b=variant_b_name,
            metric=self._metric,
            score_a=score_a,
            score_b=score_b,
            winner=winner,
            delta=delta,
            details_a={"latency_seconds": latency_a, "exact_match": exact_a, "keyword_match": kw_a},
            details_b={"latency_seconds": latency_b, "exact_match": exact_b, "keyword_match": kw_b},
        )

    @staticmethod
    def _normalize(s: str) -> str:
        import re
        return " ".join(re.split(r"\s+", (s or "").lower().strip()))
