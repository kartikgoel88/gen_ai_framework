"""Queue API: enqueue batch RAG, batch bills, agent runs; poll task status."""

from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.framework.config import get_settings_dep, FrameworkSettings
from src.framework.queue import is_queue_available

router = APIRouter(prefix="/tasks/queue", tags=["queue"])


class QueueRAGRequest(BaseModel):
    question: str
    top_k: int = 4


class QueueBillsRequest(BaseModel):
    bill_texts: list[str]
    policy_text: str
    file_names: Optional[list[str]] = None


class QueueAgentRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = None


def _ensure_queue(settings: FrameworkSettings) -> None:
    from fastapi import HTTPException
    if not is_queue_available(getattr(settings, "CELERY_BROKER_URL", None)):
        raise HTTPException(503, "Task queue not configured. Set CELERY_BROKER_URL and install celery[redis].")


@router.post("/rag")
def queue_rag(
    body: QueueRAGRequest,
    settings: FrameworkSettings = Depends(get_settings_dep),
):
    """Enqueue a RAG query. Returns task_id; poll GET /tasks/queue/status/{task_id} for result."""
    _ensure_queue(settings)
    from src.framework.queue.tasks import batch_rag_task
    t = batch_rag_task.delay(body.question, top_k=body.top_k)
    return {"task_id": t.id, "status": "queued"}


@router.post("/bills")
def queue_bills(
    body: QueueBillsRequest,
    settings: FrameworkSettings = Depends(get_settings_dep),
):
    """Enqueue batch bills processing. Returns task_id; poll status for result."""
    _ensure_queue(settings)
    from src.framework.queue.tasks import batch_bills_task
    t = batch_bills_task.delay(
        bill_texts=body.bill_texts,
        policy_text=body.policy_text,
        file_names=body.file_names,
    )
    return {"task_id": t.id, "status": "queued"}


@router.post("/agent")
def queue_agent(
    body: QueueAgentRequest,
    settings: FrameworkSettings = Depends(get_settings_dep),
):
    """Enqueue agent run. Returns task_id; poll status for result."""
    _ensure_queue(settings)
    from src.framework.queue.tasks import agent_run_task
    t = agent_run_task.delay(body.message, system_prompt=body.system_prompt)
    return {"task_id": t.id, "status": "queued"}


@router.get("/status/{task_id}")
def queue_status(
    task_id: str,
    settings: FrameworkSettings = Depends(get_settings_dep),
) -> dict[str, Any]:
    """Get task status and result (when ready). Requires CELERY_RESULT_BACKEND."""
    _ensure_queue(settings)
    from src.framework.queue.app import get_celery_app
    from celery.result import AsyncResult
    app = get_celery_app(
        broker_url=getattr(settings, "CELERY_BROKER_URL", None),
        result_backend=getattr(settings, "CELERY_RESULT_BACKEND", None),
    )
    if app is None:
        return {"task_id": task_id, "status": "unknown", "error": "Queue not configured"}
    r = AsyncResult(task_id, app=app)
    out: dict[str, Any] = {"task_id": task_id, "status": r.status}
    if r.ready():
        if r.successful():
            out["result"] = r.result
        else:
            out["error"] = str(r.result) if r.result else "Task failed"
    return out
