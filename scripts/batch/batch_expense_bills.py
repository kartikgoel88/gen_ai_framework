#!/usr/bin/env python3
"""
Batch expense bills: process policy + employee folders (or ZIP) in one script.

All logic lives here. Uses only framework (config, LLM, document/OCR) and
batch validations/schemas. No runner or BatchExpenseService calls.

Edit CONFIG below and run: python -m scripts.batch.batch_expense_bills
"""

from __future__ import annotations

import json
import logging
import re
import sys
import zipfile
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

# Suppress PyTorch MPS warning on Apple Silicon
warnings.filterwarnings("ignore", message=".*pin_memory.*not supported on MPS", category=UserWarning)

# Project root
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# -----------------------------------------------------------------------------
# CONFIG — edit for your run
# -----------------------------------------------------------------------------
POLICY_PATH = Path("/resources/policy/company_policy.pdf")
# Optional: prepended to policy text so approval rules (mandatory vs optional fields) are applied first
POLICY_RULES_PATH = Path("resources/policy/approval_rules.txt")
FOLDERS_DIR = Path("data/uploads/batch_bills/processed_inputs/meal")  # or None
ZIP_PATH = None
CLIENT_ADDRESSES_PATH = None
OUTPUT_PATH = None
MODEL_OVERRIDE = None
RETRIES = 3
DRY_RUN = False
# -----------------------------------------------------------------------------

# Bill types and allowed file extensions
BILL_TYPE_CAB = "cab"
BILL_TYPE_MEALS = "meals"
BILL_TYPE_UNKNOWN = "unknown"
BILL_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".docx", ".txt"}


# =============================================================================
# PATH HELPERS
# =============================================================================

def _resolve_path(p: Path | None) -> Path | None:
    if p is None:
        return None
    path = Path(p)
    return path if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def get_output_path() -> Path:
    from src.framework.config import get_settings
    if OUTPUT_PATH is not None:
        return _resolve_path(OUTPUT_PATH)  # type: ignore
    return (_PROJECT_ROOT / get_settings().OUTPUT_DIR / "batch" / "results.json").resolve()


def get_folder_paths(
    folders_dir: Path | None,
    zip_path: Path | None,
    upload_dir: Path,
) -> list[Path]:
    """List employee folder paths: from directory or from extracted ZIP."""
    if folders_dir is not None:
        folders_dir = Path(folders_dir)
        if not folders_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {folders_dir}")
        return sorted([folders_dir / d.name for d in folders_dir.iterdir() if d.is_dir()])
    if zip_path is not None:
        zip_path = Path(zip_path)
        if not zip_path.is_file():
            raise FileNotFoundError(f"ZIP not found: {zip_path}")
        extract_root = upload_dir / "batch_runner_extract"
        root = extract_root / zip_path.stem
        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)
        return sorted([root / d.name for d in root.iterdir() if d.is_dir()])
    return []


def load_json_file(path: Path | None) -> dict:
    if path is None or not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


# =============================================================================
# POLICY
# =============================================================================

def load_policy_text(policy_path: Path, doc_processor: Any, ocr_processor: Any) -> str:
    """Extract policy text from PDF/TXT/DOCX. Uses doc_processor for docs, OCR for images."""
    from src.framework.documents import IMAGE_EXTENSIONS
    path = Path(policy_path)
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS and ocr_processor:
        result = ocr_processor.extract(path)
        text = result.text if not result.error else ""
    else:
        result = doc_processor.extract(path)
        text = result.text if not result.error else ""
    return (text or "").strip() or "Default: approve if amount is reasonable; reject if amount missing or suspicious."


def extract_policy_to_json(policy_text: str, llm: Any) -> dict[str, Any]:
    """Extract structured policy fields (categories, limits, conditions) as JSON."""
    if not (policy_text or "").strip():
        return {}
    prompt = """Parse the following expense policy into a structured JSON object. Extract:

1. policy_categories: array of strings (e.g. ["cab", "meals", "client_location_allowance", "fuel_two_wheeler", "fuel_four_wheeler"])
2. amount_limits: object mapping category or description to max amount in INR (e.g. {"client_location_allowance": 6000, "meals": 125, "cab_per_month": 6000, "fuel_per_km_two_wheeler": 5, "fuel_per_km_four_wheeler": 10})
3. max_amounts: object with keys like "per_month_cab", "per_day_meals", "client_allowance_per_month" and numeric values in INR
4. approval_conditions: array of strings (e.g. "Submit bills by 30th", "Receipts mandatory", "Pro-rata by days")
5. effective_date: string (e.g. "June 2025") if mentioned
6. submission_rules: array of strings (deadlines, who to submit to, etc.)
7. other: object for any other key fields (reimbursement_delay, shared_cab_rule, etc.)

Policy text:
"""
    prompt += (policy_text or "")[:5000]
    prompt += "\n\nReturn only valid JSON with the above structure. Use null for missing fields. No markdown."
    try:
        out = llm.invoke_structured(prompt)
        if isinstance(out, dict):
            return out
        return {}
    except Exception:
        return {}


# =============================================================================
# EMPLOYEE CONTEXT (from folder name)
# =============================================================================

def employee_context_from_foldername(folder_name: str) -> dict[str, Any]:
    """Parse folder name e.g. emp_id_emp_name_oct_tesco -> emp_id, emp_name, emp_month, client."""
    parts = folder_name.strip().split("_")
    return {
        "emp_id": parts[0] if len(parts) > 0 else "",
        "emp_name": parts[1] if len(parts) > 1 else "",
        "emp_month": parts[2] if len(parts) > 2 else "",
        "client": "_".join(parts[3:]) if len(parts) > 3 else None,
    }


def list_bill_files(folder_path: Path) -> list[Path]:
    """List bill files (by extension) in folder; non-recursive."""
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        return []
    return sorted([folder_path / f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in BILL_EXTENSIONS])


# =============================================================================
# TEXT EXTRACTION
# =============================================================================

def extract_text_from_file(
    path: Path,
    doc_processor: Any,
    ocr_processor: Any,
) -> str:
    """Extract text from a bill file. Images use OCR; PDF/DOCX use doc_processor."""
    from src.framework.documents import IMAGE_EXTENSIONS
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS and ocr_processor:
        result = ocr_processor.extract(path)
        return result.text if not result.error else ""
    result = doc_processor.extract(path)
    return result.text if not result.error else ""


# =============================================================================
# BILL CLASSIFICATION
# =============================================================================

def classify_bill_type(text: str, file_name: str, llm: Any) -> str:
    """Classify bill as cab, meals, or unknown. Uses keywords then optional LLM."""
    lower = (text + " " + file_name).lower()
    if any(k in lower for k in ("cab", "taxi", "uber", "lyft", "ride", "transport")):
        return BILL_TYPE_CAB
    if any(k in lower for k in ("meal", "food", "restaurant", "dining", "lunch", "dinner", "breakfast", "catering")):
        return BILL_TYPE_MEALS
    try:
        prompt = f"""Given this bill/receipt text (or filename), say ONLY one word: cab, meals, or unknown.
Text/filename: {text[:1500]} {file_name}
One word:"""
        out = llm.invoke(prompt).strip().lower()
        if "cab" in out:
            return BILL_TYPE_CAB
        if "meal" in out:
            return BILL_TYPE_MEALS
    except Exception:
        pass
    return BILL_TYPE_UNKNOWN


# =============================================================================
# BILL FIELD EXTRACTION (LLM) + RUPEE NORMALIZATION
# =============================================================================

def _normalize_amount(extracted: dict[str, Any]) -> None:
    """Normalize amount: handle ₹, Rs., INR, commas. Ensure amount is a number and currency INR when Indian."""
    if not isinstance(extracted, dict):
        return
    raw_amount = extracted.get("amount")
    raw_currency = (extracted.get("currency") or "").strip()
    # If amount is string, strip rupee symbols and parse
    if isinstance(raw_amount, str):
        s = raw_amount.strip()
        # Remove rupee symbols (₹ ₨), Rs., INR, Rupees, and OCR glitches like { in place of ₹
        s = re.sub(r"[\u20b9\u20a8]|Rs\.?|INR|Rupees?|[,{}\s]", "", s, flags=re.I).strip()
        try:
            extracted["amount"] = float(s)
        except ValueError:
            # First try: number with optional dot (e.g. 135.0 or 7108.66)
            m = re.search(r"[\d,]+\.?\d*", s)
            if m:
                try:
                    extracted["amount"] = float(m.group(0).replace(",", ""))
                except ValueError:
                    pass
            if extracted.get("amount") is None:
                m = re.search(r"[\d.]+", s)
                if m:
                    try:
                        extracted["amount"] = float(m.group(0).replace(",", ""))
                    except ValueError:
                        pass
    elif isinstance(raw_amount, (int, float)):
        extracted["amount"] = float(raw_amount)
    # Default currency to INR if not set and amount looks Indian (or we always assume INR for this batch)
    if not raw_currency and extracted.get("amount") is not None:
        extracted["currency"] = "INR"


def extract_bill_fields(text: str, bill_type: str, llm: Any) -> dict[str, Any]:
    """Extract amount, date, vendor, etc. from bill text via LLM. Normalizes rupee amounts."""
    if not text.strip():
        return {"raw": "", "error": "No text extracted"}
    extra_cab = "\n- rider_name: string or null\n- pickup_address: string or null\n- drop_address: string or null\n- id: string or null"
    extra_meal = "\n- buyer_name: string or null\n- id: string or null"
    extra = extra_cab if bill_type == BILL_TYPE_CAB else (extra_meal if bill_type == BILL_TYPE_MEALS else "")
    prompt = f"""Extract key fields from this {bill_type} bill/receipt. Return a JSON object with these keys (use null if not found):

- amount: number only (total payable). The bill may show amount with Indian Rupee symbols (₹, Rs., INR, Rupees) or in words — extract the numeric value only (e.g. 135 or 6000). Ignore currency symbols; just the number.
- currency: string (e.g. INR if Indian Rupees)
- date: string (prefer DD/MM/YYYY)
- vendor: string (merchant/vendor name)
- description: string (brief)
- items: array of strings{extra}

Bill text:
{text[:3000]}

Return only valid JSON, no markdown."""
    try:
        result = llm.invoke_structured(prompt)
        if isinstance(result, dict):
            _normalize_amount(result)
            return result
        return {"raw": text[:500], "error": "Could not parse"}
    except Exception as e:
        return {"raw": text[:500], "error": str(e)}


# =============================================================================
# VALIDATION (uses batch validations)
# =============================================================================

def validate_bill(
    bill_type: str,
    extracted: dict,
    file_name: str,
    employee_context: dict,
    client_addresses: dict,
) -> tuple[dict, str]:
    """Run rule-based validation; return (validation_dict, bill_id)."""
    from src.clients.batch import validations as val
    bill = {**extracted, "file_name": file_name, "filename": file_name}
    bill["client"] = employee_context.get("client")
    bill["emp_name"] = employee_context.get("emp_name")
    bill["emp_month"] = employee_context.get("emp_month")
    emp_name = employee_context.get("emp_name")
    emp_month = employee_context.get("emp_month")
    if bill_type == BILL_TYPE_CAB:
        v = val.validate_ride(bill, client_addresses or {}, emp_name=emp_name, emp_month=emp_month)
    elif bill_type == BILL_TYPE_MEALS:
        v = val.validate_meal(bill, emp_name=emp_name, emp_month=emp_month)
    else:
        v = {"is_valid": True, "month_match": True, "name_match": True, "address_match": True}
    bill_id = bill.get("id") or f"MANUAL-{file_name}-{id(bill)}"
    extracted["id"] = bill_id
    return v, bill_id


# =============================================================================
# POLICY EVALUATION (LLM)
# =============================================================================

def evaluate_policy(
    bill_type: str,
    extracted: dict,
    policy_text: str,
    llm: Any,
) -> tuple[str, str]:
    """Decide APPROVE/REJECT and reason using policy text and bill data."""
    if not (policy_text or "").strip():
        policy_text = "Default: approve if amount is reasonable; reject if amount missing or suspicious."
    prompt = f"""You are an expense approver. Apply the following admin policy to this bill and decide APPROVE or REJECT. Then give a short reason.

Admin policy text:
{policy_text[:2000]}

Bill type: {bill_type}
Extracted bill data (JSON):
{json.dumps(extracted, default=str)}

Respond with a JSON object with exactly two keys:
- "decision": "APPROVED" or "REJECTED"
- "reason": "one or two sentence reason"
Return only valid JSON."""
    try:
        result = llm.invoke_structured(prompt)
        decision = (result.get("decision") or "REJECTED").upper()
        reason = (result.get("reason") or "No reason provided").strip() or "No reason provided"
        if "APPROVE" in decision and "REJECT" not in decision:
            decision = "APPROVED"
        else:
            decision = "REJECTED"
        return decision, reason
    except Exception as e:
        return "REJECTED", f"Policy evaluation failed: {e}"


# =============================================================================
# PROCESS ONE BILL
# =============================================================================

def process_one_bill(
    path: Path,
    policy_text: str,
    employee_context: dict,
    client_addresses: dict,
    doc_processor: Any,
    ocr_processor: Any,
    llm: Any,
) -> dict[str, Any]:
    """Extract, classify, extract fields, validate, evaluate policy. Return one result dict."""
    if not path.exists():
        return {
            "file_name": path.name,
            "bill_type": BILL_TYPE_UNKNOWN,
            "extracted": {},
            "decision": "REJECTED",
            "reason": "File not found",
            "validation": {},
            "employee_context": employee_context,
            "bill_id": None,
        }
    text = extract_text_from_file(path, doc_processor, ocr_processor)
    bill_type = classify_bill_type(text, path.name, llm)
    extracted = extract_bill_fields(text, bill_type, llm)
    validation, bill_id = validate_bill(bill_type, extracted, path.name, employee_context, client_addresses)
    decision, reason = evaluate_policy(bill_type, extracted, policy_text, llm)
    return {
        "file_name": path.name,
        "bill_type": bill_type,
        "extracted": extracted,
        "decision": decision,
        "reason": reason,
        "validation": validation,
        "employee_context": employee_context,
        "bill_id": bill_id,
    }


# =============================================================================
# PROCESS ONE FOLDER (employee)
# =============================================================================

def process_employee_folder(
    folder_path: Path,
    policy_text: str,
    client_addresses: dict,
    doc_processor: Any,
    ocr_processor: Any,
    llm: Any,
) -> tuple[dict, list[dict]]:
    """Process all bill files in one employee folder. Returns (employee_context, results)."""
    folder_path = Path(folder_path)
    employee_context = employee_context_from_foldername(folder_path.name)
    bill_paths = list_bill_files(folder_path)
    if not bill_paths:
        return employee_context, []
    results = []
    for path in bill_paths:
        r = process_one_bill(
            path, policy_text, employee_context, client_addresses,
            doc_processor, ocr_processor, llm,
        )
        results.append(r)
    return employee_context, results


# =============================================================================
# PROCESS ALL FOLDERS
# =============================================================================

def process_all_folders(
    folder_paths: list[Path],
    policy_text: str,
    client_addresses: dict,
    doc_processor: Any,
    ocr_processor: Any,
    llm: Any,
) -> dict[str, Any]:
    """Process each employee folder; return {folders: [...], summary: {approved, rejected, total}}."""
    folders_out = []
    total_approved = 0
    total_rejected = 0
    for folder_path in folder_paths:
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            continue
        employee_context, results = process_employee_folder(
            folder_path, policy_text, client_addresses,
            doc_processor, ocr_processor, llm,
        )
        approved = sum(1 for r in results if (r.get("decision") or "").upper() == "APPROVED")
        rejected = len(results) - approved
        total_approved += approved
        total_rejected += rejected
        folders_out.append({
            "folder_name": folder_path.name,
            "employee_context": employee_context,
            "results": results,
            "summary": {"approved": approved, "rejected": rejected, "total": len(results)},
        })
    return {
        "folders": folders_out,
        "summary": {"approved": total_approved, "rejected": total_rejected, "total": total_approved + total_rejected},
    }


# =============================================================================
# RUNNER (orchestration + retry)
# =============================================================================

from scripts.batch.base import BaseBatchRunner, ExitCode, retry_with_backoff
from scripts.batch.common import apply_framework_overrides, setup_logging


class ExpenseBillsBatchRunner(BaseBatchRunner):
    """Orchestrates config, validation, and process_all_folders with retry."""

    def run(self) -> int:
        args = self.parse_args()
        setup_logging(level="INFO", log_file=None)
        self.log = logging.getLogger("scripts.batch.ExpenseBillsBatchRunner")
        try:
            self.validate_args(args)
        except (ValueError, FileNotFoundError, NotADirectoryError) as e:
            self.log.error("Validation failed: %s", e)
            return ExitCode.USAGE_OR_VALIDATION
        if getattr(args, "dry_run", False):
            self.dry_run(args)
            return ExitCode.SUCCESS
        try:
            out = self.run_batch(args)
        except Exception as e:
            self.log.exception("Batch run failed: %s", e)
            return ExitCode.TOTAL_FAILURE
        success = out.get("success_count", 0)
        failure = out.get("failure_count", 0)
        if failure == 0:
            return ExitCode.SUCCESS
        if success > 0:
            return ExitCode.PARTIAL_FAILURE
        return ExitCode.TOTAL_FAILURE

    def parse_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            policy=_resolve_path(POLICY_PATH),
            policy_rules=_resolve_path(POLICY_RULES_PATH) if POLICY_RULES_PATH else None,
            folders=_resolve_path(FOLDERS_DIR) if FOLDERS_DIR else None,
            zip=_resolve_path(ZIP_PATH) if ZIP_PATH else None,
            client_addresses=_resolve_path(CLIENT_ADDRESSES_PATH) if CLIENT_ADDRESSES_PATH else None,
            output=get_output_path(),
            model=MODEL_OVERRIDE,
            retries=RETRIES,
            dry_run=DRY_RUN,
            log_level="INFO",
            log_file=None,
        )

    def validate_args(self, args: SimpleNamespace) -> None:
        if args.folders is None and args.zip is None:
            raise ValueError("Set either FOLDERS_DIR or ZIP_PATH in CONFIG.")
        if not args.policy.exists():
            raise FileNotFoundError(f"Policy file not found: {args.policy}")
        if args.folders is not None and not args.folders.is_dir():
            raise NotADirectoryError(f"Not a directory: {args.folders}")
        if args.zip is not None and not args.zip.is_file():
            raise FileNotFoundError(f"ZIP not found: {args.zip}")
        if args.client_addresses is not None and not args.client_addresses.exists():
            raise FileNotFoundError(f"Client addresses file not found: {args.client_addresses}")
        args.output.parent.mkdir(parents=True, exist_ok=True)

    def dry_run(self, args: SimpleNamespace) -> None:
        if args.folders is not None:
            subdirs = [d for d in args.folders.iterdir() if d.is_dir()]
            print(f"[batch-expense-bills] Dry run: would process {len(subdirs)} folders. Output: {args.output}")
            for d in sorted(subdirs)[:10]:
                print(f"  {d.name}")
            if len(subdirs) > 10:
                print(f"  ... and {len(subdirs) - 10} more")
        else:
            print(f"[batch-expense-bills] Dry run: would extract ZIP and process employee folders. Output: {args.output}")

    def run_batch(self, args: SimpleNamespace) -> dict[str, Any]:
        from src.framework.config import get_settings
        from src.framework.api.deps_llm import get_llm
        from src.framework.documents import OcrProcessor
        from src.framework.documents.processor import DocumentProcessor

        if args.model is not None:
            apply_framework_overrides(model=args.model)
        settings = get_settings()
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)

        ocr = OcrProcessor(pdf_dpi=300, pdf_min_text_len=10)
        doc_processor = DocumentProcessor(upload_dir=str(upload_dir), pdf_processor=ocr)
        llm = get_llm(settings)
        client_addresses = load_json_file(args.client_addresses)

        print("[batch-expense-bills] Starting.")
        print(f"  Policy: {args.policy}")
        print(f"  Input:  {'folders ' + str(args.folders) if args.folders else 'zip ' + str(args.zip)}")
        print(f"  Output: {args.output}")
        print(f"  Config: UPLOAD_DIR={settings.UPLOAD_DIR}, LLM_MODEL={settings.LLM_MODEL}, LLM_PROVIDER={settings.LLM_PROVIDER}")
        if (getattr(settings, "LLM_PROVIDER", "") or "").lower() == "local":
            print("  Note: LLM_PROVIDER=local — ensure Ollama is running: ollama serve && ollama run llama3.2")

        policy_text = load_policy_text(args.policy, doc_processor, ocr)
        if getattr(args, "policy_rules", None) and Path(args.policy_rules).exists():
            rules_text = Path(args.policy_rules).read_text(encoding="utf-8").strip()
            if rules_text:
                policy_text = f"{rules_text}\n\n{policy_text}".strip()
                print(f"[batch-expense-bills] Prepend approval rules from {args.policy_rules}")
        policy_extract = extract_policy_to_json(policy_text, llm)
        policy_extract_path = args.output.parent / "policy_extract.json"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(policy_extract_path, "w", encoding="utf-8") as f:
            json.dump(policy_extract, f, indent=2, default=str, ensure_ascii=False)
        print(f"[batch-expense-bills] Policy extracted to {policy_extract_path}")

        folder_paths = get_folder_paths(args.folders, args.zip, upload_dir)

        if args.folders is not None:
            subdirs = sorted([d for d in args.folders.iterdir() if d.is_dir()])
            print(f"[batch-expense-bills] Found {len(subdirs)} employee subfolder(s).")
            for sub in subdirs[:5]:
                bills = list_bill_files(sub)
                print(f"  '{sub.name}': {len(bills)} bill(s) {[f.name for f in bills[:3]]}{'...' if len(bills) > 3 else ''}")
            if len(subdirs) > 5:
                print(f"  ... and {len(subdirs) - 5} more")

        if not folder_paths:
            result = {"folders": [], "summary": {"approved": 0, "rejected": 0, "total": 0}}
        else:
            print("[batch-expense-bills] Running batch processing...")
            result = retry_with_backoff(
                lambda: process_all_folders(
                    folder_paths, policy_text, client_addresses,
                    doc_processor, ocr, llm,
                ),
                max_attempts=args.retries,
                initial_delay=2.0,
                max_delay=120.0,
                log=self.log,
                is_retryable=lambda e: "rate limit" in str(e).lower() or "429" in str(e),
            )

        if policy_text:
            print("[batch-expense-bills] Policy preview:")
            print("  ---")
            for line in policy_text.strip().splitlines()[:25]:
                print(f"  {line}")
            print("  ---")

        summary = result.get("summary", {})
        total = summary.get("total", 0)
        approved = summary.get("approved", 0)
        rejected = summary.get("rejected", 0)
        for fo in result.get("folders", []):
            name = fo.get("folder_name", "?")
            results_list = fo.get("results", [])
            print(f"[batch-expense-bills] Folder '{name}': {len(results_list)} bill(s) processed.")
            for r in results_list:
                fname = r.get("file_name", "?")
                decision = (r.get("decision") or "").strip() or "—"
                reason = (r.get("reason") or "").strip()
                err = (r.get("extracted") or {}).get("error") if isinstance(r.get("extracted"), dict) else None
                print(f"    {fname}: {decision}" + (f" — {reason}" if reason else "") + (f" ({err})" if err else ""))
        print(f"[batch-expense-bills] Total: Approved {approved}, Rejected {rejected}, Total {total}.")
        print(f"[batch-expense-bills] Results written to {args.output}")

        args.output.write_text(json.dumps(result, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        return {"success_count": total, "failure_count": 0, "errors": [], "result": result, "summary": summary}


def main() -> int:
    if MODEL_OVERRIDE is not None:
        apply_framework_overrides(model=MODEL_OVERRIDE)
    runner = ExpenseBillsBatchRunner(description="Batch expense bills (self-contained).")
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
