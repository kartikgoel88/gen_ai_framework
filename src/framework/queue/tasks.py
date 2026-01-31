"""Celery tasks for batch RAG, batch bills, and long agent runs."""

from typing import Any, Optional

# Celery app is configured in app.py; tasks are auto-discovered when broker is set.
# We use a lazy app getter so we don't require celery at import time.


def _get_app():
    from .app import get_celery_app
    from ..config import get_settings
    s = get_settings()
    return get_celery_app(
        broker_url=getattr(s, "CELERY_BROKER_URL", None),
        result_backend=getattr(s, "CELERY_RESULT_BACKEND", None),
    )


def _task(fn):
    """Decorator that binds fn to Celery app when app is available."""
    app = _get_app()
    if app is not None:
        return app.task(bind=True)(fn)
    return fn


@_task
def batch_rag_task(self, question: str, top_k: int = 4) -> dict[str, Any]:
    """Run RAG query in background. Returns {question, answer}."""
    from ..api.deps import get_rag, get_llm
    from ..config import get_settings
    s = get_settings()
    rag = get_rag(s)
    llm = get_llm(s)
    answer = rag.query(question, llm_client=llm, top_k=top_k)
    return {"question": question, "answer": answer}


@_task
def batch_bills_task(
    self,
    bill_texts: list[str],
    policy_text: str,
    file_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Run batch expense (bills + policy) in background. Returns list of results."""
    from src.clients.batch.service import BatchExpenseService
    from ..config import get_settings
    from ..api.deps import get_llm, get_document_processor, get_pdf_ocr_processor, get_ocr_processor
    s = get_settings()
    llm = get_llm(s)
    doc_processor = get_document_processor(s, get_pdf_ocr_processor())
    ocr_processor = get_ocr_processor()
    service = BatchExpenseService(llm=llm, doc_processor=doc_processor, ocr_processor=ocr_processor)
    results = service.process_bills_from_texts(
        bill_texts=bill_texts, policy_text=policy_text, file_names=file_names
    )
    return {"results": results, "policy_preview": policy_text[:200]}


@_task
def agent_run_task(self, message: str, system_prompt: Optional[str] = None) -> dict[str, Any]:
    """Run agent (RAG + MCP tools) in background. Returns {output}."""
    from ..agents.langchain_agent import LangChainReActAgent
    from ..agents.tools import build_framework_tools
    from ..api.deps import get_rag, get_mcp_client
    from ..config import get_settings
    from langchain_openai import ChatOpenAI
    s = get_settings()
    rag = get_rag(s)
    mcp = get_mcp_client(s)
    tools = build_framework_tools(rag_client=rag, mcp_client=mcp)
    llm = ChatOpenAI(model=s.LLM_MODEL, temperature=s.TEMPERATURE, openai_api_key=s.OPENAI_API_KEY)
    agent = LangChainReActAgent(llm=llm, tools=tools, system_prompt=system_prompt, verbose=False)
    output = agent.invoke(message, system_prompt=system_prompt)
    return {"output": output}
