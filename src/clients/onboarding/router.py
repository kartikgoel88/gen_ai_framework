"""Onboarding client API routes."""

from fastapi import APIRouter, Depends, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional

from src.framework.api.deps import get_llm, get_rag, get_document_processor
from src.framework.llm.base import LLMClient
from src.framework.rag.base import RAGClient
from src.framework.documents.processor import DocumentProcessor

router = APIRouter(prefix="/onboarding", tags=["onboarding"])


class WelcomeEmailRequest(BaseModel):
    employee_name: str
    employee_email: str
    start_date: str
    role: Optional[str] = None


class WelcomeEmailResponse(BaseModel):
    subject: str
    body: str


@router.post("/welcome-email", response_model=WelcomeEmailResponse)
def generate_welcome_email(
    req: WelcomeEmailRequest,
    llm: LLMClient = Depends(get_llm),
):
    """Generate a personalized onboarding welcome email using the LLM."""
    prompt = f"""Generate a short professional welcome email for a new employee.
Name: {req.employee_name}
Email: {req.employee_email}
Start date: {req.start_date}
Role: {req.role or 'Not specified'}
Return a JSON object with exactly two keys: "subject" (string) and "body" (string, plain text)."""
    result = llm.invoke_structured(prompt)
    return WelcomeEmailResponse(
        subject=result.get("subject", "Welcome to the team!"),
        body=result.get("body", ""),
    )


@router.post("/ingest-handbook")
async def ingest_handbook(
    file: UploadFile = File(...),
    doc_processor: DocumentProcessor = Depends(get_document_processor),
    rag: RAGClient = Depends(get_rag),
):
    """Upload employee handbook (or any doc) and ingest into RAG for onboarding Q&A."""
    content = await file.read()
    path = doc_processor.save_upload(content, file.filename or "handbook.pdf")
    result = doc_processor.extract(path)
    if result.error:
        return {"ok": False, "error": result.error}
    rag.add_documents([result.text], metadatas=[{"source": file.filename, **result.metadata}])
    return {"ok": True, "message": "Handbook ingested for RAG"}


@router.get("/ask")
def ask_onboarding(
    q: str,
    rag: RAGClient = Depends(get_rag),
    llm: LLMClient = Depends(get_llm),
):
    """Ask a question about onboarding (RAG over ingested docs)."""
    answer = rag.query(q, llm_client=llm)
    return {"question": q, "answer": answer}
