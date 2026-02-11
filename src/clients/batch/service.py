"""Batch expense service: parse bills and evaluate against policy using framework."""

import json
from pathlib import Path
from typing import Any, Optional

from src.framework.llm.base import LLMClient
from src.framework.documents import IMAGE_EXTENSIONS, OcrProcessor
from src.framework.documents.processor import DocumentProcessor

from . import validations as val
from .schemas import EmployeeContext, PolicySection

# Bill type constants
BILL_TYPE_CAB = "cab"
BILL_TYPE_MEALS = "meals"
BILL_TYPE_UNKNOWN = "unknown"

# Bill file extensions (for folder listing)
BILL_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".docx", ".txt"}


class BatchExpenseService:
    """Parse cab/meals bills and decide approve/reject based on admin policy."""

    def __init__(
        self,
        llm: LLMClient,
        doc_processor: DocumentProcessor,
        ocr_processor: Optional[OcrProcessor] = None,
    ):
        self._llm = llm
        self._doc = doc_processor
        self._ocr = ocr_processor

    def extract_text_from_file(self, file_path: Path) -> str:
        """Extract text from a bill file. All document handling (including PDF with OCR fallback) goes through DocumentProcessor; images use OCR processor."""
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS and self._ocr:
            result = self._ocr.extract(path)
            return result.text if not result.error else ""
        result = self._doc.extract(path)
        return result.text if not result.error else ""

    # -------------------------------------------------------------------------
    # Folder-based: one folder per employee (food + cab bills); policy once.
    # -------------------------------------------------------------------------
    @staticmethod
    def extract_employee_context_from_foldername(folder_name: str) -> EmployeeContext:
        """Parse folder name e.g. IIIPL-1011_smitha_oct_tesco -> emp_id, emp_name, emp_month, client."""
        parts = folder_name.strip().split("_")
        emp_id = parts[0] if len(parts) > 0 else ""
        emp_name = parts[1] if len(parts) > 1 else ""
        emp_month = parts[2] if len(parts) > 2 else ""
        client = "_".join(parts[3:]) if len(parts) > 3 else ""
        return EmployeeContext(emp_id=emp_id, emp_name=emp_name, emp_month=emp_month, client=client or None)

    @staticmethod
    def list_bill_files(folder_path: Path) -> list[Path]:
        """List bill files (PDF, images, DOCX, TXT) in folder; non-recursive."""
        path = Path(folder_path)
        if not path.is_dir():
            return []
        return [path / f for f in path.iterdir() if f.is_file() and f.suffix.lower() in BILL_EXTENSIONS]

    def process_employee_folder(
        self,
        folder_path: Path,
        policy_text: str,
        client_addresses: dict[str, list[str]],
        policy_json: Optional[dict[str, Any]] = None,
    ) -> tuple[EmployeeContext, list[dict[str, Any]]]:
        """
        Process one employee folder: parse employee context from folder name,
        list bill files, run grouped processing (meal + cab). Policy is reused via policy_json.
        Returns (employee_context, list of bill results).
        """
        folder_path = Path(folder_path)
        folder_name = folder_path.name
        employee_context = self.extract_employee_context_from_foldername(folder_name)
        bill_paths = self.list_bill_files(folder_path)
        if not bill_paths:
            return employee_context, []
        results = self.process_bills_grouped(
            file_paths=bill_paths,
            policy_text=policy_text,
            employee_context=employee_context,
            client_addresses=client_addresses,
            policy_json=policy_json,
        )
        return employee_context, results

    def process_folders(
        self,
        folder_paths: list[Path],
        policy_text: str,
        client_addresses: Optional[dict[str, list[str]]] = None,
    ) -> dict[str, Any]:
        """
        Process multiple employee folders. Policy is parsed once and reused for all folders.
        Each folder = one employee (folder name = emp_id_emp_name_month_client), contains food + cab bills.
        Returns dict: folders=[{folder_name, employee_context, results, summary}], summary={approved, rejected, total}.
        """
        client_addresses = client_addresses or {}
        policy_json = self.parse_policy_to_json(policy_text)
        folders_out: list[dict[str, Any]] = []
        total_approved = 0
        total_rejected = 0
        for folder_path in folder_paths:
            folder_path = Path(folder_path)
            if not folder_path.is_dir():
                continue
            employee_context, results = self.process_employee_folder(
                folder_path,
                policy_text=policy_text,
                client_addresses=client_addresses,
                policy_json=policy_json,
            )
            approved = sum(1 for r in results if (r.get("decision") or "").upper() == "APPROVED")
            rejected = len(results) - approved
            total_approved += approved
            total_rejected += rejected
            folders_out.append({
                "folder_name": folder_path.name,
                "employee_context": employee_context.model_dump(),
                "results": results,
                "summary": {"approved": approved, "rejected": rejected, "total": len(results)},
            })
        return {
            "folders": folders_out,
            "summary": {"approved": total_approved, "rejected": total_rejected, "total": total_approved + total_rejected},
        }

    # -------------------------------------------------------------------------
    # Policy: parse to JSON and pass with decision prompt (not RAG).
    # For very large policy docs, RAG could be used later to retrieve relevant
    # policy chunks; for now we pass the full parsed JSON with the decision.
    # -------------------------------------------------------------------------
    def parse_policy_to_json(self, policy_text: str) -> dict[str, Any]:
        """Parse policy text to structured JSON via LLM (section, category, amount_valid, additional_conditions). Pass with decision prompt, not RAG."""
        if not (policy_text or "").strip():
            return {}
        prompt = """Parse the following expense policy text into a structured JSON object with a "sections" array.
Each section must have:
- "section": string (policy heading, e.g. "Client Location Allowance", "Meal Allowance")
- "category": string (expense category: cab, meals, client_location, fuel_two_wheeler, fuel_four_wheeler, or similar)
- "amount_valid": number or null (max/valid amount in currency units, e.g. 6000, 125; null if not specified)
- "additional_conditions": array of strings or null (other conditions, e.g. "Submit bills by 30th", "Receipts required", "Pro-rata by days")

Extract every distinct policy rule (meal limits, cab/travel limits, fuel per km, client allowance, submission rules) as separate sections.
Use null for missing fields. Return only valid JSON with a single root object containing "sections". No markdown.

Policy text:
"""
        prompt += (policy_text or "")[:4000]
        try:
            out = self._llm.invoke_structured(prompt)
            if not isinstance(out, dict):
                return {}
            sections = out.get("sections")
            if not isinstance(sections, list):
                return out
            # Validate and normalize to PolicySection schema (section, category, amount_valid, additional_conditions)
            validated = []
            for s in sections:
                if not isinstance(s, dict):
                    continue
                try:
                    validated.append(PolicySection.model_validate(s).model_dump())
                except Exception:
                    validated.append({k: v for k, v in s.items() if k in ("section", "category", "amount_valid", "additional_conditions")})
            return {"sections": validated}
        except Exception:
            return {}

    def classify_bill_type(self, text: str, file_name: str) -> str:
        """Classify bill as cab or meals from content/filename."""
        lower = (text + " " + file_name).lower()
        if any(k in lower for k in ("cab", "taxi", "uber", "lyft", "ride", "transport")):
            return BILL_TYPE_CAB
        if any(k in lower for k in ("meal", "food", "restaurant", "dining", "lunch", "dinner", "breakfast", "catering")):
            return BILL_TYPE_MEALS
        # Ask LLM if ambiguous
        try:
            prompt = f"""Given this bill/receipt text (or filename), say ONLY one word: cab, meals, or unknown.
Text/filename: {text[:1500]} {file_name}
One word:"""
            out = self._llm.invoke(prompt).strip().lower()
            if "cab" in out:
                return BILL_TYPE_CAB
            if "meal" in out:
                return BILL_TYPE_MEALS
        except Exception:
            pass
        return BILL_TYPE_UNKNOWN

    def extract_bill_fields(self, text: str, bill_type: str) -> dict[str, Any]:
        """Extract structured fields (amount, date, vendor, etc.) from bill text."""
        if not text.strip():
            return {"raw": "", "error": "No text extracted"}
        # Cab: include rider_name, pickup_address, drop_address for validations
        extra_cab = "\n- rider_name: string or null\n- pickup_address: string or null\n- drop_address: string or null\n- id: string or null (bill/trip id)"
        extra_meal = "\n- buyer_name: string or null\n- id: string or null"
        extra = extra_cab if bill_type == BILL_TYPE_CAB else (extra_meal if bill_type == BILL_TYPE_MEALS else "")
        prompt = f"""Extract key fields from this {bill_type} bill/receipt. Return a JSON object with these keys (use null if not found):
- amount: number (total amount)
- currency: string
- date: string (prefer DD/MM/YYYY)
- vendor: string (merchant/vendor name)
- description: string (brief)
- items: array of strings (line items if any){extra}

Bill text:
{text[:3000]}

Return only valid JSON, no markdown."""
        try:
            result = self._llm.invoke_structured(prompt)
            if isinstance(result, dict):
                return result
            return {"raw": text[:500], "error": "Could not parse"}
        except Exception as e:
            return {"raw": text[:500], "error": str(e)}

    def _validate_bill(
        self,
        bill_type: str,
        extracted: dict[str, Any],
        file_name: str,
        employee_context: Optional[EmployeeContext],
        client_addresses: dict[str, list[str]],
    ) -> tuple[dict[str, Any], str]:
        """Run rule-based validation; attach id if missing. Returns (validation_dict, bill_id)."""
        emp_name = employee_context.emp_name if employee_context else None
        emp_month = employee_context.emp_month if employee_context else None
        bill = {**extracted, "file_name": file_name, "filename": file_name}
        if employee_context:
            bill["client"] = employee_context.client
            bill["emp_name"] = employee_context.emp_name
            bill["emp_month"] = employee_context.emp_month
        if bill_type == BILL_TYPE_CAB:
            v = val.validate_ride(
                bill,
                client_addresses or {},
                emp_name=emp_name,
                emp_month=emp_month,
            )
        elif bill_type == BILL_TYPE_MEALS:
            v = val.validate_meal(
                bill,
                emp_name=emp_name,
                emp_month=emp_month,
            )
        else:
            v = {"is_valid": True, "month_match": True, "name_match": True, "address_match": True}
        bill_id = bill.get("id") or f"MANUAL-{file_name}-{id(bill)}"
        extracted["id"] = bill_id
        return v, bill_id

    def evaluate_policy(
        self,
        bill_type: str,
        extracted: dict[str, Any],
        policy_text: str,
        policy_json: Optional[dict[str, Any]] = None,
    ) -> tuple[str, str]:
        """Decide approve/reject and reason. Uses policy_text; if policy_json provided, pass it with prompt (not RAG)."""
        if not (policy_text or "").strip() and not policy_json:
            policy_text = "Default: approve if amount is reasonable and date/vendor are present; reject if amount is missing or suspicious."
        policy_block = ""
        if policy_json:
            policy_block = f"\nStructured policy (JSON):\n{json.dumps(policy_json, default=str)}\n\n"
        if policy_text and policy_text.strip():
            policy_block += f"Admin policy text:\n{policy_text[:2000]}"
        prompt = f"""You are an expense approver. Apply the following admin policy to this bill and decide APPROVE or REJECT. Then give a short reason.
{policy_block}

Bill type: {bill_type}
Extracted bill data (JSON):
{json.dumps(extracted, default=str)}

Respond with a JSON object with exactly two keys:
- "decision": "APPROVED" or "REJECTED"
- "reason": "one or two sentence reason"
Return only valid JSON."""
        try:
            result = self._llm.invoke_structured(prompt)
            decision = (result.get("decision") or "REJECTED").upper()
            reason = (result.get("reason") or "No reason provided").strip()
            if not reason:
                reason = "No reason provided"
            if "APPROVE" in decision and "REJECT" not in decision:
                decision = "APPROVED"
            else:
                decision = "REJECTED"
            return decision, reason
        except Exception as e:
            return "REJECTED", f"Policy evaluation failed: {e}"

    def evaluate_policy_groups(
        self,
        policy_text: str,
        policy_json: dict[str, Any],
        groups_data: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """One LLM call: policy JSON + groups → list of group decisions. Policy passed with prompt (not RAG)."""
        prompt = """You are an expense approver. For each group below, compare daily_total (meals) or monthly_total (cab) to the policy and output APPROVE or REJECT with reasons.

Structured policy (JSON):
"""
        prompt += json.dumps(policy_json or {}, default=str)
        prompt += "\n\nGroups (each has valid_bill_ids, invalid_bill_ids, daily_total or monthly_total):\n"
        prompt += json.dumps(groups_data, default=str, indent=2)
        prompt += """

Return a JSON array. Each element: {"decision": "APPROVE" or "REJECT", "employee_id": "...", "employee_name": "...", "category": "...", "valid_bill_ids": [...], "invalid_bill_ids": [...], "reasons": ["..."]}
One element per group, in the same order as the groups. Return only valid JSON array."""
        try:
            out = self._llm.invoke_structured(prompt)
            if isinstance(out, list):
                return out
            if isinstance(out, dict) and "groups" in out:
                return out["groups"]
            return []
        except Exception:
            return []

    def process_bills(
        self,
        file_paths: list[Path],
        policy_text: str,
        employee_context: Optional[EmployeeContext] = None,
        client_addresses: Optional[dict[str, list[str]]] = None,
        use_policy_json: bool = True,
    ) -> list[dict[str, Any]]:
        """Process multiple bill files: extract, classify, extract fields, validate, evaluate policy."""
        policy_json = self.parse_policy_to_json(policy_text) if use_policy_json else None
        client_addresses = client_addresses or {}
        results = []
        for path in file_paths:
            path = Path(path)
            if not path.exists():
                results.append({
                    "file_name": path.name,
                    "bill_type": BILL_TYPE_UNKNOWN,
                    "extracted": {},
                    "decision": "REJECTED",
                    "reason": "File not found",
                    "validation": {},
                    "employee_context": employee_context.model_dump() if employee_context else None,
                    "bill_id": None,
                })
                continue
            text = self.extract_text_from_file(path)
            bill_type = self.classify_bill_type(text, path.name)
            extracted = self.extract_bill_fields(text, bill_type)
            validation, bill_id = self._validate_bill(
                bill_type, extracted, path.name, employee_context, client_addresses
            )
            decision, reason = self.evaluate_policy(
                bill_type, extracted, policy_text, policy_json=policy_json
            )
            emp_ctx_dict = employee_context.model_dump() if employee_context else None
            results.append({
                "file_name": path.name,
                "bill_type": bill_type,
                "extracted": extracted,
                "decision": decision,
                "reason": reason,
                "validation": validation,
                "employee_context": emp_ctx_dict,
                "bill_id": bill_id,
            })
        return results

    def process_bills_grouped(
        self,
        file_paths: list[Path],
        policy_text: str,
        employee_context: Optional[EmployeeContext] = None,
        client_addresses: Optional[dict[str, list[str]]] = None,
        policy_json: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Process bills with grouping: by (employee, category) for cab, by (employee, category, date) for meals.
        Policy is parsed to JSON and passed with decision prompt (not RAG). One decision per group.
        Pass policy_json to reuse pre-parsed policy (e.g. when processing multiple folders)."""
        if policy_json is None:
            policy_json = self.parse_policy_to_json(policy_text)
        # Per-bill: extract, classify, extract fields, validate
        client_addresses = client_addresses or {}
        bills_with_meta: list[dict[str, Any]] = []
        emp_id = (employee_context.emp_id or "").strip() if employee_context else ""
        emp_name = (employee_context.emp_name or "").strip() if employee_context else ""

        for path in file_paths:
            path = Path(path)
            if not path.exists():
                continue
            text = self.extract_text_from_file(path)
            bill_type = self.classify_bill_type(text, path.name)
            extracted = self.extract_bill_fields(text, bill_type)
            validation, bill_id = self._validate_bill(
                bill_type, extracted, path.name, employee_context, client_addresses
            )
            extracted["id"] = bill_id
            amount = float(extracted.get("amount") or 0)
            date_val = extracted.get("date") or ""
            bills_with_meta.append({
                "file_name": path.name,
                "bill_type": bill_type,
                "extracted": extracted,
                "validation": validation,
                "employee_context": employee_context.model_dump() if employee_context else None,
                "bill_id": bill_id,
                "amount": amount,
                "date": date_val,
            })

        # Group: meals by (emp_id, emp_name, category, date); cab by (emp_id, emp_name, category)
        groups_map: dict[tuple, dict[str, Any]] = {}
        for b in bills_with_meta:
            cat = b["bill_type"] if b["bill_type"] in (BILL_TYPE_CAB, BILL_TYPE_MEALS) else "unknown"
            ctx = b.get("employee_context") or {}
            key_emp = (emp_id or ctx.get("emp_id") or "", emp_name or ctx.get("emp_name") or "")
            is_valid = b["validation"].get("is_valid", True)
            bid = b["bill_id"]
            amount = b["amount"]
            date_val = b["date"]

            if cat == BILL_TYPE_MEALS:
                gkey = (*key_emp, cat, date_val or "unknown")
            else:
                gkey = (*key_emp, cat, None)

            if gkey not in groups_map:
                groups_map[gkey] = {
                    "employee_id": key_emp[0],
                    "employee_name": key_emp[1],
                    "category": cat,
                    "date": gkey[3] if len(gkey) > 3 else None,
                    "valid_bill_ids": [],
                    "invalid_bill_ids": [],
                    "daily_total": 0.0,
                    "monthly_total": 0.0,
                }
            g = groups_map[gkey]
            if is_valid:
                g["valid_bill_ids"].append(bid)
                if cat == BILL_TYPE_MEALS:
                    g["daily_total"] = (g["daily_total"] or 0) + amount
                else:
                    g["monthly_total"] = (g["monthly_total"] or 0) + amount
            else:
                g["invalid_bill_ids"].append(bid)

        groups_data = list(groups_map.values())
        group_decisions = self.evaluate_policy_groups(policy_text, policy_json, groups_data)

        # Map decision back to each bill
        decision_by_bill_id: dict[str, tuple[str, str]] = {}
        for i, g in enumerate(groups_data):
            dec = group_decisions[i] if i < len(group_decisions) else {}
            decision = (dec.get("decision") or "REJECT").upper()
            decision = "APPROVED" if "APPROVE" in decision and "REJECT" not in decision else "REJECTED"
            reasons = dec.get("reasons") or [dec.get("reason", "")]
            reason = "; ".join(str(r).strip() for r in (reasons if isinstance(reasons, list) else [reasons]) if str(r).strip())
            if not reason:
                reason = "No reason provided by approver"
            for bid in g["valid_bill_ids"] + g["invalid_bill_ids"]:
                decision_by_bill_id[bid] = (decision, reason)

        results = []
        for b in bills_with_meta:
            dec, reason = decision_by_bill_id.get(b["bill_id"], ("REJECTED", "No group decision"))
            if not (reason or "").strip():
                reason = "No reason provided"
            results.append({
                "file_name": b["file_name"],
                "bill_type": b["bill_type"],
                "extracted": b["extracted"],
                "decision": dec,
                "reason": reason,
                "validation": b["validation"],
                "employee_context": b["employee_context"],
                "bill_id": b["bill_id"],
            })
        return results

    def process_bills_from_texts(
        self,
        bill_texts: list[str],
        policy_text: str,
        file_names: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Process bills from raw text (no file I/O). For use in task queue. Returns list of {file_name, bill_type, extracted, decision, reason, validation, bill_id}."""
        file_names = file_names or [f"bill_{i}.txt" for i in range(len(bill_texts))]
        policy_json = self.parse_policy_to_json(policy_text)
        results = []
        for i, text in enumerate(bill_texts):
            fname = file_names[i] if i < len(file_names) else f"bill_{i}.txt"
            bill_type = self.classify_bill_type(text, fname)
            extracted = self.extract_bill_fields(text, bill_type)
            validation, bill_id = self._validate_bill(
                bill_type, extracted, fname, None, {}
            )
            extracted["id"] = bill_id
            decision, reason = self.evaluate_policy(
                bill_type, extracted, policy_text, policy_json=policy_json
            )
            results.append({
                "file_name": fname,
                "bill_type": bill_type,
                "extracted": extracted,
                "decision": decision,
                "reason": reason,
                "validation": validation,
                "employee_context": None,
                "bill_id": bill_id,
            })
        return results
