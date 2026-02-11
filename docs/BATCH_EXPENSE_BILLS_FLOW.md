# Batch Expense Bills — High-Level Flow

This document describes the end-to-end flow of the **batch expense bills** script (`scripts/batch/batch_expense_bills.py`): from policy and employee folders (or ZIP) to a results JSON and a structured policy extract.

---

## Overview

The script processes expense bills (cab, meals, etc.) against an admin policy. It:

1. Loads a **policy** document (PDF, image, or text) and optionally extracts structured fields to JSON.
2. Discovers **employee folders** (from a directory or a ZIP).
3. For each folder, finds **bill files** (PDF, images, DOCX, TXT), extracts text, classifies type, extracts fields, validates, and runs a policy **approve/reject** decision.
4. Writes **results** (per-folder, per-bill) to `results.json` and **policy extract** to `policy_extract.json`.

**Text extraction uses OCR for PDFs and images** so that scanned PDFs and photo-based bills are supported.

---

## Flow Diagram (High Level)

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Policy file     │     │  Employee        │     │  Config             │
│  (PDF/image/txt) │     │  folders or ZIP  │     │  (UPLOAD_DIR, LLM)  │
└────────┬─────────┘     └────────┬─────────┘     └──────────┬──────────┘
         │                        │                         │
         ▼                        ▼                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. SETUP                                                            │
│     • Resolve paths (policy, folders, output)                         │
│     • Create DocumentProcessor (with OCR for PDFs) + OcrProcessor    │
│     • Create LLM client (from framework config)                      │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. POLICY                                                           │
│     • Load policy text: PDF/image → OCR or doc processor; TXT → read │
│     • Extract policy to JSON (categories, limits, conditions)         │
│     • Write policy_extract.json                                      │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. FOLDERS                                                          │
│     • List employee folders (from directory or extract ZIP)           │
│     • For each folder: list bill files (.pdf, .jpg, .png, .docx, …)  │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. PER BILL (each file in each folder)                              │
│     • Extract text (see “Text extraction & OCR” below)               │
│     • Classify bill type (cab / meals / unknown)                     │
│     • Extract fields (amount, date, vendor, …) — normalize rupee      │
│     • Validate (month, name, address for cab)                        │
│     • Evaluate policy (LLM: APPROVE / REJECT + reason)               │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. OUTPUT                                                           │
│     • results.json: folders[], summary (approved/rejected/total)       │
│     • policy_extract.json: policy_categories, amount_limits, etc.    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Text Extraction and OCR

**OCR is used for PDFs and for image files** so that scanned policy PDFs and photo/screenshot bills are handled correctly.

### Policy and bill files

| Input type | How text is obtained |
|------------|------------------------|
| **PDF** | `DocumentProcessor` is built with `pdf_processor=OcrProcessor`. For each PDF it first runs **OCR** (PyMuPDF + Tesseract fallback for low-text/scanned pages). If OCR returns no text, it falls back to native PyPDF text. So **image-only or scanned PDFs use OCR**. |
| **Images** (.png, .jpg, .jpeg, .webp, .bmp, .tiff) | **OCR only**: `OcrProcessor` (EasyOCR by default, or Tesseract if configured). |
| **DOCX / TXT** | `DocumentProcessor` native extraction (no OCR). |

### Components

- **DocumentProcessor** (framework): Single entry for PDF, DOCX, Excel, TXT. For PDFs, when `pdf_processor` is set to an `OcrProcessor`, it delegates to OCR first, then falls back to native PDF text if needed.
- **OcrProcessor** (framework):  
  - **PDFs**: PyMuPDF to get text; for pages with little or no text, uses **Tesseract OCR** (scanned/image PDFs).  
  - **Images**: **EasyOCR** (default) or Tesseract.

So in the batch expense flow, **all PDFs are eligible for OCR** (and use it when they are image-based), and **all image bill/policy files use OCR**.

---

## Steps in More Detail

1. **Config & paths**  
   CONFIG in the script (or env) sets policy path, folders dir or ZIP, output path. Paths are resolved relative to project root.

2. **Policy load**  
   Policy file is read with `DocumentProcessor` + `OcrProcessor`: PDFs and images use OCR as above; TXT/DOCX use normal extraction. Policy text is then passed to the LLM for structured extraction.

3. **Policy extract**  
   LLM parses policy into JSON: `policy_categories`, `amount_limits`, `max_amounts`, `approval_conditions`, `effective_date`, `submission_rules`, etc. Written to `data/output/batch/policy_extract.json` (or next to `results.json`).

4. **Folder discovery**  
   Either list subdirectories of `FOLDERS_DIR` (one subdir = one employee) or extract a ZIP and list subdirs there.

5. **Per-bill pipeline**  
   For each bill file in each employee folder:
   - **Extract text**: PDF → DocumentProcessor (OCR path for image PDFs); images → OCR.
   - **Classify**: cab / meals / unknown (keywords + optional LLM).
   - **Extract fields**: LLM returns amount, date, vendor, etc.; **rupee symbols (₹, Rs., INR) are normalized** to a numeric amount and INR currency.
   - **Validate**: Rule-based (month, name, address for cab) using batch validations.
   - **Evaluate**: LLM applies policy text to extracted bill and returns APPROVE/REJECT and reason.

6. **Output**  
   - `results.json`: one entry per folder with `results` (per-bill) and `summary` (approved/rejected/total).  
   - `policy_extract.json`: structured policy fields.

---

## Output Files

| File | Description |
|------|-------------|
| `results.json` | Per-folder and per-bill results; summary counts (approved, rejected, total). |
| `policy_extract.json` | Structured policy: categories, amount limits, max amounts, conditions, dates, submission rules. |

Default location: `data/output/batch/` (overridable via CONFIG / `OUTPUT_PATH`).

---

## Controlling mandatory fields (policy text only)

The approve/reject decision is driven by the **admin policy text** you load (e.g. from `company_policy.pdf` or a `.txt` file). The code does not hard-code which fields are mandatory; the LLM follows whatever rules you put in that policy. To avoid rejections like “lacks date, rider name, pro-rata compliance”, **reframe your policy text** so it explicitly defines what is required and what is optional.

### 1. Define mandatory vs optional in the policy

Add a short “Approval rules” or “Mandatory fields” block at the top (or in a prominent place) of your policy document. For example:

```text
Approval rules for expense approver:
- Mandatory for APPROVE: only (1) amount in INR present and within policy limits, and (2) bill type matches a policy category (e.g. cab, meals). Nothing else is mandatory.
- Optional (do not reject solely for absence): rider name, exact date, base location, pro-rata details, vendor name. If these are missing or unclear, still APPROVE provided amount and category are acceptable.
- REJECT only if: amount is missing or above the stated limit for that category, or bill type is not covered by policy, or amount is clearly suspicious. Do not REJECT for missing optional details.
```

Adjust the list of “mandatory” vs “optional” to match what you want (e.g. if you do want date to be mandatory, say “Mandatory: amount and date” and remove date from the optional list).

### 2. Be explicit about what “reject” means

To avoid the model inventing extra requirements, state rejection criteria narrowly:

```text
Reject only when:
1. Amount is missing or cannot be determined, or
2. Amount exceeds the policy limit for that expense category, or
3. Expense category is not allowed by policy.
Do not reject for: missing rider name, missing or unclear date, unclear pro-rata or base location, or lack of other non-mandatory details.
```

### 3. Example policy snippet (cab)

```text
Cab / transport:
- Max ₹6000 per month per employee.
- Mandatory for approval: amount (INR) present and ≤ 6000; bill identifiable as cab/transport.
- Optional: rider name, pick-up/drop address, exact date, base location. If absent or unclear, still approve if amount and category are OK.
- Reject only if amount missing, or amount > 6000, or not a cab/transport bill.
```

You can paste the same rules into the UI “Policy text” or put them in your policy PDF/text file. The LLM will then treat only the listed items as mandatory and stop rejecting bills solely for missing date, rider name, or pro-rata/base-location wording.

---

## Dependencies for OCR

- **Tesseract** (for PDF OCR and optional image OCR): install e.g. `brew install tesseract` (macOS), `apt install tesseract-ocr` (Linux).  
- **EasyOCR** (for images): used by `OcrProcessor` for image files; installed with the project.  
- **PyMuPDF** (fitz): used for PDF rendering before Tesseract; installed with the project.

---

## See Also

- `scripts/batch/README.md` — How to run the script, CONFIG, and other batch scripts.
- `scripts/batch/batch_expense_bills.py` — Inline comments and section headers for code-level detail.
