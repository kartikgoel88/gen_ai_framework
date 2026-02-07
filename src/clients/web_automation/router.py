"""Web automation client API: run automation with URL, fill fields, upload files."""

import json
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.framework.web_automation import (
    WebAutomationClient,
    RunResult,
    AutomationStep,
    UploadStep,
)

router = APIRouter(prefix="/web-automation", tags=["web-automation"])


class RunSpecBody(BaseModel):
    """JSON body for run (no file uploads). Use POST /run with form + files when steps include upload."""

    url: str = Field(..., description="Page URL to open")
    steps: List[dict] = Field(..., description="List of step objects: fill, click, wait (no upload)")


def _parse_steps(steps_data: list) -> list[AutomationStep]:
    """Parse step dicts into typed AutomationStep list."""
    return [AutomationStep.model_validate(s) for s in steps_data]


@router.post("/run-json", response_model=RunResult)
async def run_automation_json(body: RunSpecBody):
    """
    Run web automation with JSON only (no file uploads). Use for fill + click + wait only.
    For steps that include file upload, use POST /run with form (spec) and files.
    """
    try:
        steps = _parse_steps(body.steps)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid step in 'steps': {e}") from e
    for step in steps:
        if step.action == "upload":
            raise HTTPException(
                status_code=422,
                detail="Use POST /run with form and files for upload steps.",
            )
    client = WebAutomationClient(headless=True, timeout_ms=30_000)
    return await client.run(url=body.url, steps=steps)


@router.post("/run", response_model=RunResult)
async def run_automation(
    spec: str = Form(
        ...,
        description='JSON: {"url": "https://...", "steps": [{"action": "fill", "selector": "#id", "value": "..."}, {"action": "upload", "selector": "input[type=file]", "file_index": 0}, {"action": "click", "selector": "button[type=submit]"}]}',
    ),
    files: List[UploadFile] = File(
        default_factory=list,
        description="Files for upload steps (order matches file_index: first file = 0).",
    ),
):
    """
    Run web automation: open URL and run steps (fill fields, upload files, click, wait).

    - **spec**: JSON with `url` and `steps`. Steps:
      - `fill`: `selector`, `value`
      - `upload`: `selector`, `file_index` (0-based index into `files`)
      - `click`: `selector`
      - `wait`: `selector` (wait until visible) or `timeout_ms`
    - **files**: Optional list of files; required when any step has action `upload`.
    """
    try:
        data = json.loads(spec)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid spec JSON: {e}") from e

    url = data.get("url")
    if not url or not isinstance(url, str):
        raise HTTPException(status_code=422, detail="spec must include a non-empty 'url' string")

    steps_data = data.get("steps")
    if not isinstance(steps_data, list):
        raise HTTPException(status_code=422, detail="spec must include 'steps' as a list")

    try:
        steps = _parse_steps(steps_data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid step in 'steps': {e}") from e

    # Resolve upload steps: save uploaded files to temp paths and set file_path on steps
    temp_paths: list[Path] = []
    try:
        steps_with_paths: list[AutomationStep] = []
        for step in steps:
            if step.action == "upload":
                idx = step.file_index
                if idx < 0 or idx >= len(files):
                    raise HTTPException(
                        status_code=422,
                        detail=f"Upload step references file_index {idx} but only {len(files)} file(s) provided.",
                    )
                upload = files[idx]
                suffix = Path(upload.filename or "upload").suffix or ".bin"
                fd, path = tempfile.mkstemp(suffix=suffix)
                temp_paths.append(Path(path))
                content = await upload.read()
                with open(path, "wb") as f:
                    f.write(content)
                # Build new step with file_path set (UploadStep is immutable)
                step_with_path = UploadStep(
                    action="upload",
                    selector=step.selector,
                    file_index=step.file_index,
                    file_path=path,
                )
                steps_with_paths.append(step_with_path)
            else:
                steps_with_paths.append(step)

        client = WebAutomationClient(headless=True, timeout_ms=30_000)
        result = await client.run(url=url, steps=steps_with_paths)
        return result
    finally:
        for p in temp_paths:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
