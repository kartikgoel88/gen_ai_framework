"""Batch client API: process cab/meals bills against admin policy."""

import json
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException

from src.framework.api.deps import (
    get_llm,
    get_document_processor,
    get_ocr_processor,
)
from src.framework.llm.base import LLMClient
from src.framework.documents.processor import DocumentProcessor
from src.framework.documents import OcrProcessor

from .service import BatchExpenseService
from .schemas import (
    ProcessResultItem,
    BatchSummary,
    BatchProcessResponse,
    ProcessFoldersResponse,
    FolderResult,
    EmployeeContext,
)

router = APIRouter(prefix="/batch", tags=["batch"])


def get_batch_service(
    llm: LLMClient = Depends(get_llm),
    doc_processor: DocumentProcessor = Depends(get_document_processor),
    ocr_processor: OcrProcessor = Depends(get_ocr_processor),
) -> BatchExpenseService:
    """Build batch expense service from framework components. Document handling (including PDF with OCR) goes through DocumentProcessor."""
    return BatchExpenseService(
        llm=llm,
        doc_processor=doc_processor,
        ocr_processor=ocr_processor,
    )


def _parse_employee_context(s: Optional[str]):
    """Parse optional JSON form field to EmployeeContext or None."""
    if not s or not s.strip():
        return None
    try:
        d = json.loads(s)
        return EmployeeContext(**d) if d else None
    except Exception:
        return None


def _parse_client_addresses(s: Optional[str]) -> dict:
    """Parse optional JSON form field to {client: [addresses]}."""
    if not s or not s.strip():
        return {}
    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _extract_zip_to_dir(zip_path: Path, extract_root: Path) -> list[Path]:
    """Extract zip to extract_root; return list of paths to top-level directories (one per employee folder)."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)
    return [extract_root / d.name for d in extract_root.iterdir() if d.is_dir()]


@router.post("/process-folders", response_model=ProcessFoldersResponse)
async def process_folders(
    folders_zip: UploadFile = File(..., description="ZIP containing one folder per employee (folder name = emp_id_emp_name_month_client); each folder has food + cab bill files."),
    policy_file: Optional[UploadFile] = File(None, description="Policy document (PDF/TXT/DOCX). Optional if policy_text is provided."),
    policy_text: Optional[str] = Form(None, description="Policy as text. Optional if policy_file is provided. Policy is applied once for all folders."),
    client_addresses: Optional[str] = Form(None, description='Optional JSON: {"TESCO": ["addr1"], "AMEX": ["addr2"]} for cab address validation.'),
    doc_processor: DocumentProcessor = Depends(get_document_processor),
    batch_service: BatchExpenseService = Depends(get_batch_service),
):
    """
    Process employee folders: one folder per employee with food + cab bills; policy document is separate and applied once for all.
    Upload a ZIP whose top-level entries are folders named e.g. IIIPL-1011_smitha_oct_tesco; each folder contains meal and cab bill files.
    Provide policy once (file or text); it is parsed once and reused for every folder.
    """
    if not policy_text and not policy_file:
        raise HTTPException(400, "Provide policy_file or policy_text")

    policy_source = None
    if policy_file:
        policy_content = await policy_file.read()
        policy_path = doc_processor.save_upload(policy_content, policy_file.filename or "policy.pdf")
        policy_text = batch_service.extract_text_from_file(policy_path) or policy_text or ""
        policy_source = "file"
    policy_text = (policy_text or "").strip() or "Default: approve if amount is reasonable; reject if amount missing or suspicious."
    if not policy_source:
        policy_source = "text"

    zip_content = await folders_zip.read()
    zip_path = doc_processor.save_upload(zip_content, folders_zip.filename or "folders.zip")
    extract_root = Path(doc_processor.upload_dir) / "batch_folders" / zip_path.stem
    extract_root.mkdir(parents=True, exist_ok=True)
    folder_paths = _extract_zip_to_dir(Path(zip_path), extract_root)

    clients = _parse_client_addresses(client_addresses)
    out = batch_service.process_folders(folder_paths, policy_text=policy_text, client_addresses=clients)

    folder_results = [
        FolderResult(
            folder_name=f["folder_name"],
            employee_context=f["employee_context"],
            results=[ProcessResultItem(**r) for r in f["results"]],
            summary=BatchSummary(**f["summary"]),
        )
        for f in out["folders"]
    ]
    summary = BatchSummary(**out["summary"])
    return ProcessFoldersResponse(
        policy_source=policy_source,
        folders=folder_results,
        summary=summary,
    )


@router.post("/process", response_model=BatchProcessResponse)
async def process_bills(
    files: list[UploadFile] = File(..., description="Cab or meals bill files (PDF, images, DOCX, etc.)."),
    policy_file: Optional[UploadFile] = File(None, description="Policy document (PDF/TXT/DOCX). Optional if policy_text provided."),
    policy_text: Optional[str] = Form(None, description="Policy as text. Optional if policy_file provided."),
    employee_context: Optional[str] = Form(
        None,
        description='Optional JSON: {"emp_id", "emp_name", "emp_month", "client"} for validations and grouping.',
    ),
    client_addresses: Optional[str] = Form(
        None,
        description='Optional JSON: {"TESCO": ["addr1"], "AMEX": ["addr2"]} for cab address validation.',
    ),
    doc_processor: DocumentProcessor = Depends(get_document_processor),
    batch_service: BatchExpenseService = Depends(get_batch_service),
):
    """
    Process a flat list of bills. Provide policy once (file or text).
    When employee_context is provided, uses group-level decisions (meal/cab by category/date); otherwise per-bill decision.
    """
    if not files:
        return BatchProcessResponse(results=[], summary=BatchSummary(approved=0, rejected=0, total=0))
    if not policy_text and not policy_file:
        raise HTTPException(400, "Provide policy_file or policy_text")

    if policy_file:
        policy_content = await policy_file.read()
        policy_path = doc_processor.save_upload(policy_content, policy_file.filename or "policy.pdf")
        policy_text = batch_service.extract_text_from_file(policy_path) or policy_text or ""
    policy_text = (policy_text or "").strip() or "Default: approve if amount is reasonable; reject if amount missing or suspicious."

    paths = [doc_processor.save_upload(await f.read(), f.filename or "bill") for f in files]
    emp_ctx = _parse_employee_context(employee_context)
    clients = _parse_client_addresses(client_addresses)

    if emp_ctx:
        results = batch_service.process_bills_grouped(paths, policy_text=policy_text, employee_context=emp_ctx, client_addresses=clients)
    else:
        results = batch_service.process_bills(paths, policy_text=policy_text, client_addresses=clients)

    approved = sum(1 for r in results if (r.get("decision") or "").upper() == "APPROVED")
    rejected = len(results) - approved
    return BatchProcessResponse(
        results=[ProcessResultItem(**r) for r in results],
        summary=BatchSummary(approved=approved, rejected=rejected, total=len(results)),
    )
