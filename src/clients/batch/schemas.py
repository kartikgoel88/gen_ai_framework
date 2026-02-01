"""Pydantic schemas for batch expense (admin bills) flow."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# --- Employee context (optional; enriches bills and drives validations) ---
class EmployeeContext(BaseModel):
    """Employee metadata: from folder naming or API input. Used for validations and grouping."""

    emp_id: Optional[str] = Field(None, description="Employee ID")
    emp_name: Optional[str] = Field(None, description="Employee name (for name match validation)")
    emp_month: Optional[str] = Field(None, description="Expected month (e.g. oct, nov)")
    client: Optional[str] = Field(None, description="Client code (e.g. TESCO, AMEX) for address validation")


# --- Extracted bill (LLM output) ---
class BillExtracted(BaseModel):
    """Structured fields extracted from a bill."""

    amount: Optional[float] = Field(None, description="Total amount")
    currency: Optional[str] = None
    date: Optional[str] = None
    vendor: Optional[str] = None
    description: Optional[str] = None
    items: Optional[list[str]] = None
    raw: Optional[str] = None
    error: Optional[str] = None

    model_config = ConfigDict(extra="allow")  # Allow LLM to return extra fields


# --- Policy decision (LLM output) ---
class PolicyDecision(BaseModel):
    """Approve/reject decision and reason."""

    decision: str = Field(..., description="APPROVED or REJECTED")
    reason: str = Field(..., description="Short reason")


# --- Single bill result ---
class ProcessResultItem(BaseModel):
    """Result of processing one bill."""

    file_name: str
    bill_type: str = Field(..., description="cab | meals | unknown")
    extracted: dict[str, Any] = Field(default_factory=dict)
    decision: str = Field(..., description="APPROVED | REJECTED")
    reason: str = ""
    validation: Optional[dict[str, Any]] = Field(None, description="Rule-based validation (month, name, address)")
    employee_context: Optional[dict[str, Any]] = Field(None, description="emp_id, emp_name, emp_month, client")
    bill_id: Optional[str] = Field(None, description="Stable id (for grouping)")


# --- Batch summary ---
class BatchSummary(BaseModel):
    """Summary counts for a batch run."""

    approved: int = 0
    rejected: int = 0
    total: int = 0


# --- Full batch response ---
class BatchProcessResponse(BaseModel):
    """Response for batch process endpoint."""

    results: list[ProcessResultItem] = Field(default_factory=list)
    summary: BatchSummary = Field(default_factory=BatchSummary)


# --- Folder-based: one folder per employee (food + cab bills) ---
class FolderResult(BaseModel):
    """Result for one employee folder (meal + cab bills)."""

    folder_name: str = Field(..., description="e.g. IIIPL-1011_smitha_oct_tesco")
    employee_context: Optional[dict[str, Any]] = Field(None, description="emp_id, emp_name, emp_month, client from folder name")
    results: list[ProcessResultItem] = Field(default_factory=list)
    summary: BatchSummary = Field(default_factory=BatchSummary)


class ProcessFoldersResponse(BaseModel):
    """Response for process-folders: policy applied once; one result per employee folder."""

    policy_source: Optional[str] = Field(None, description="'file' or 'text'")
    folders: list[FolderResult] = Field(default_factory=list)
    summary: BatchSummary = Field(default_factory=BatchSummary)


# --- Test / file-based run output (for writing to output folder) ---
class BatchRunOutput(BaseModel):
    """Output written to output folder: same as API response plus optional metadata."""

    results: list[ProcessResultItem] = Field(default_factory=list)
    summary: BatchSummary = Field(default_factory=BatchSummary)
    input_policy_path: Optional[str] = None
    input_bill_paths: list[str] = Field(default_factory=list)
