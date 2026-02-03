"""Multi-model evaluation example.

Evaluates the same eval dataset across multiple models (e.g. gpt-4 vs gpt-3.5)
using EvalHarness and evaluate_multiple_models. Use this pattern to compare
providers or model sizes on exact match, keyword match, and latency.
"""

from src.framework.config import get_settings
from src.framework.llm.registry import LLMProviderRegistry
from src.framework.observability.eval import (
    EvalDatasetItem,
    EvalHarness,
    evaluate_multiple_models,
)


def main():
    settings = get_settings()
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        print("Set OPENAI_API_KEY to run this example.")
        return

    # Define models to compare: (model_id, provider, model name)
    model_configs = [
        ("gpt-4-turbo", "openai", "gpt-4-turbo-preview"),
        ("gpt-3.5-turbo", "openai", "gpt-3.5-turbo"),
    ]

    # API key per provider (use the one you need for your providers)
    def api_key_for(provider: str):
        p = provider.lower()
        if p == "grok":
            return settings.XAI_API_KEY
        if p == "gemini":
            return settings.GOOGLE_API_KEY
        if p == "huggingface":
            return settings.HUGGINGFACE_API_KEY
        return settings.OPENAI_API_KEY

    # Build (model_id, LLMClient) list without using the cached get_llm
    models = []
    for mid, provider, model_name in model_configs:
        key = api_key_for(provider)
        if not key:
            print(f"Skipping {mid}: no API key for provider {provider}")
            continue
        llm = LLMProviderRegistry.create(
            provider=provider,
            api_key=key,
            model=model_name,
            temperature=settings.TEMPERATURE,
        )
        models.append((mid, llm))

    if not models:
        print("No models configured. Check API keys and model_configs.")
        return

    # In-memory eval dataset (or load from file with EvalHarness.load_dataset)
    items: list[EvalDatasetItem] = [
        EvalDatasetItem(
            question="What is 2 + 2?",
            expected_answer="4",
        ),
        EvalDatasetItem(
            question="What color is the sky on a clear day?",
            expected_keywords=["blue"],
        ),
        EvalDatasetItem(
            question="Name a programming language.",
            expected_keywords=["Python", "Java", "JavaScript", "C++"],
        ),
    ]

    # Optional: load from JSON/JSONL instead
    # items = EvalHarness.load_dataset("path/to/eval_dataset.json")

    def progress(model_id: str, current: int, total: int) -> None:
        print(f"  {model_id}: {current}/{total}")

    print("Running evaluation for multiple models...\n")
    results = evaluate_multiple_models(
        items,
        models,
        progress_callback=progress,
    )

    # Print comparison
    print("\n" + "=" * 60)
    print("Multi-model evaluation results")
    print("=" * 60)
    for model_id, r in results.items():
        print(f"\n{model_id}:")
        print(f"  Total:        {r.total}")
        print(f"  Exact match:  {r.exact_match} ({r.exact_match_rate:.1%})")
        print(f"  Keyword match: {r.keyword_match} ({r.keyword_match_rate:.1%})")
        print(f"  Latency:      {r.latency_seconds:.2f}s")


if __name__ == "__main__":
    main()
