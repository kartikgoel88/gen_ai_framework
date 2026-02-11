"""Prompts API: versioned prompts, templates with validation, A/B testing."""

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.framework.api.deps import get_llm, get_rag
from src.framework.config import get_settings_dep, FrameworkSettings
from src.framework.llm.base import LLMClient
from src.framework.prompts.store import PromptStore, PromptVersion
from src.framework.prompts.templates import TemplateRunner
from src.framework.prompts.ab_test import ABTestRunner, ABTestResult
from src.framework.observability.eval import EvalDatasetItem

router = APIRouter(prefix="/prompts", tags=["prompts"])


def _get_prompt_store(settings: FrameworkSettings = Depends(get_settings_dep)) -> PromptStore:
    return PromptStore(base_path=getattr(settings, "PROMPTS_BASE_PATH", None))


# --- Versioned prompts ---


@router.get("/list")
def prompts_list(
    store: PromptStore = Depends(_get_prompt_store),
):
    """List prompt names (versioned prompts)."""
    return {"names": store.list_names()}


@router.get("/{name}/versions")
def prompt_versions(
    name: str,
    store: PromptStore = Depends(_get_prompt_store),
):
    """List version tags for a prompt name."""
    return {"name": name, "versions": store.list_versions(name)}


@router.get("/{name}")
def prompt_get(
    name: str,
    version: str = "v1",
    store: PromptStore = Depends(_get_prompt_store),
):
    """Get prompt body by name and version."""
    p = store.get(name, version)
    if p is None:
        return {"error": "Not found", "name": name, "version": version}
    return {"name": p.name, "version": p.version, "body": p.body}


class PromptPutRequest(BaseModel):
    body: str
    version: str = "v1"
    metadata: Optional[dict[str, Any]] = None


@router.put("/{name}")
def prompt_put(
    name: str,
    body: PromptPutRequest,
    store: PromptStore = Depends(_get_prompt_store),
):
    """Save prompt with version tag."""
    p = store.put(name, body.version, body.body, metadata=body.metadata)
    return {"name": p.name, "version": p.version}


# --- Template run (variables validated by optional Pydantic) ---


class TemplateRunRequest(BaseModel):
    template: str = Field(..., description="Prompt template with {variable} placeholders")
    inputs: dict[str, Any] = Field(default_factory=dict)
    structured: bool = Field(False, description="If true, return parsed JSON from LLM")


@router.post("/run")
def prompt_run(
    body: TemplateRunRequest,
    llm: LLMClient = Depends(get_llm),
):
    """Run a prompt template with inputs. No Pydantic validation on inputs here; use chain for that."""
    runner = TemplateRunner(llm=llm, template=body.template)
    if body.structured:
        return {"output": runner.run_structured(body.inputs)}
    return {"output": runner.run(body.inputs)}


class TemplateRunVersionedRequest(BaseModel):
    name: str
    version: str = "v1"
    inputs: dict[str, Any] = Field(default_factory=dict)
    structured: bool = False


@router.post("/run/versioned")
def prompt_run_versioned(
    body: TemplateRunVersionedRequest,
    llm: LLMClient = Depends(get_llm),
    store: PromptStore = Depends(_get_prompt_store),
):
    """Run a versioned prompt (load from store by name/version) with inputs."""
    p = store.get(body.name, body.version)
    if p is None:
        return {"error": "Prompt not found", "name": body.name, "version": body.version}
    runner = TemplateRunner(llm=llm, template=p.body)
    if body.structured:
        return {"output": runner.run_structured(body.inputs)}
    return {"output": runner.run(body.inputs)}


# --- A/B testing ---


class ABTestRequest(BaseModel):
    prompt_a: str = Field(..., description="Template with {question} placeholder")
    prompt_b: str = Field(..., description="Template with {question} placeholder")
    variant_a_name: str = "A"
    variant_b_name: str = "B"
    items: list[dict[str, Any]] = Field(
        ...,
        description="Eval items: [{question, expected_answer?, expected_keywords?}]",
    )
    metric: Optional[str] = Field("keyword_match", description="exact_match | keyword_match | latency")


@router.post("/ab-test")
def prompt_ab_test(
    body: ABTestRequest,
    llm: LLMClient = Depends(get_llm),
    rag: Any = Depends(get_rag),
    settings: FrameworkSettings = Depends(get_settings_dep),
):
    """Run two prompts on the same eval items and compare metrics."""
    items = [
        EvalDatasetItem(
            question=row.get("question", ""),
            expected_answer=row.get("expected_answer"),
            expected_keywords=row.get("expected_keywords"),
        )
        for row in body.items
    ]
    metric = body.metric or getattr(settings, "PROMPT_AB_METRIC", "keyword_match")
    runner = ABTestRunner(llm=llm, rag=rag, metric=metric)
    result = runner.run(
        prompt_a=body.prompt_a,
        prompt_b=body.prompt_b,
        items=items,
        variant_a_name=body.variant_a_name,
        variant_b_name=body.variant_b_name,
    )
    return {
        "variant_a": result.variant_a,
        "variant_b": result.variant_b,
        "metric": result.metric,
        "score_a": result.score_a,
        "score_b": result.score_b,
        "winner": result.winner,
        "delta": result.delta,
        "details_a": result.details_a,
        "details_b": result.details_b,
    }
