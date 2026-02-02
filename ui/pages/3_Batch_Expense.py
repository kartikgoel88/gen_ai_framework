"""Batch Expense: process bills against policy (uses clients.batch + framework)."""

import json
import zipfile
from pathlib import Path

import streamlit as st

from ui.services import get_batch_service, get_doc_processor, save_uploaded_file
from ui.components import parse_json_input


st.header("💼 Batch Expense Processing")
st.markdown("""
**Complete Application Example**: Process expense bills against company policy

This is a complete, production-ready application that demonstrates:
- Document processing and extraction
- LLM-based analysis and decision making
- Batch processing workflows
- Structured data extraction
- Policy compliance checking
""")

policy_source = st.radio("Policy source", ["Paste text", "Upload file"], horizontal=True)
if "policy_text" not in st.session_state:
    st.session_state["policy_text"] = ""
if policy_source == "Paste text":
    st.session_state["policy_text"] = st.text_area(
        "Policy text", value=st.session_state["policy_text"], height=120,
        placeholder="e.g. Approve meals under 500; reject if no date.", key="policy_area"
    )
else:
    policy_file = st.file_uploader("Policy file (PDF/TXT/DOCX)", type=["pdf", "txt", "docx"])
    if policy_file:
        path = save_uploaded_file(policy_file)
        if path:
            try:
                doc = get_doc_processor()
                result = doc.extract(path)
                policy_from_file = result.text or ""
                st.session_state["policy_text"] = policy_from_file
                st.session_state["policy_edit"] = policy_from_file  # sync so "Extracted policy" text area shows it
                if result.text:
                    st.success("Policy extracted. Now upload bill files below and click **Process bills** to see approval/rejection results.")
                    st.caption("If the PDF is image-only (scanned), ensure Tesseract is installed (e.g. `brew install tesseract` on macOS). You can also paste policy text above.")
                else:
                    reason = getattr(result, "error", "Unknown reason.")
                    st.warning(f"No text could be extracted from this file. **Reason:** {reason}")
                    st.caption("If the PDF is image-only (scanned), ensure Tesseract is installed (e.g. `brew install tesseract` on macOS). You can also paste policy text above.")
            except Exception as e:
                st.error(f"Extraction failed: {e}")
    # Show editable policy text (sync back to session so it's used when processing)
    edited = st.text_area(
        "Extracted policy (editable)", value=st.session_state.get("policy_text", ""), height=120, key="policy_edit"
    )
    st.session_state["policy_text"] = edited
policy_text = (st.session_state.get("policy_text") or "").strip()
if policy_text:
    st.caption("Policy is set. Upload bill files (or a ZIP of folders) below and click **Process bills** / **Process folders** to see approval/rejection results.")

# Structured policy (parsed): section, category, amount_valid, additional_conditions
if policy_text:
    st.subheader("Structured policy (extracted)")
    parse_policy = st.button("Parse policy to structure", key="parse_policy_btn")
    if parse_policy:
        with st.spinner("Parsing policy with LLM…"):
            try:
                svc = get_batch_service()
                structured = svc.parse_policy_to_json(policy_text)
                st.session_state["policy_structured"] = structured
            except Exception as e:
                st.error(f"Parse failed: {e}")
                st.session_state["policy_structured"] = None
    if st.session_state.get("policy_structured"):
        structured = st.session_state["policy_structured"]
        sections = structured.get("sections") or []
        if sections:
            rows = []
            for s in sections:
                rows.append({
                    "Section": s.get("section") or "—",
                    "Category": s.get("category") or "—",
                    "Amount valid": s.get("amount_valid") if s.get("amount_valid") is not None else None,
                    "Additional conditions": ", ".join(s.get("additional_conditions") or []) or "—",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
            with st.expander("Raw structured JSON"):
                st.json(structured)
        else:
            st.info("No sections extracted. Try rephrasing the policy or click **Parse policy to structure** again.")
    elif st.session_state.get("policy_structured") is not None and not (st.session_state["policy_structured"].get("sections")):
        st.info("No sections extracted. Try rephrasing the policy or click **Parse policy to structure** again.")

mode = st.radio("Input mode", ["Single files", "ZIP of folders"], horizontal=True)

if mode == "Single files":
    files = st.file_uploader("Bill files (PDF, images, DOCX)", type=["pdf", "png", "jpg", "jpeg", "docx", "txt"], accept_multiple_files=True)
    run_single = st.button("Process bills")
else:
    zip_file = st.file_uploader("ZIP of employee folders", type=["zip"])
    run_folders = st.button("Process folders")
    files = None
    run_single = False

client_addresses_raw = st.text_input("Client addresses (optional JSON)", placeholder='{"TESCO": ["addr1"], "AMEX": ["addr2"]}', key="client_addrs")
client_addresses, json_error = parse_json_input(client_addresses_raw, default={})
if json_error:
    st.warning(f"Invalid JSON for client addresses; ignoring. {json_error}")
if not isinstance(client_addresses, dict):
    client_addresses = {}

if run_single and files:
    if not policy_text:
        policy_text = "Default: approve if amount is reasonable; reject if amount missing or suspicious."
    try:
        svc = get_batch_service()
        doc = svc._doc
        paths = []
        for f in files:
            p = save_uploaded_file(f, "batch_bills")
            if p:
                paths.append(p)
        if not paths:
            st.error("No files saved.")
        else:
            with st.spinner("Processing…"):
                results = svc.process_bills(paths, policy_text=policy_text, client_addresses=client_addresses)
            st.subheader("Results")
            approved = sum(1 for r in results if (r.get("decision") or "").upper() == "APPROVED")
            st.metric("Approved", approved)
            st.metric("Rejected", len(results) - approved)
            for r in results:
                with st.expander(f"{r.get('file_name', '?')} — {r.get('decision', '?')}"):
                    st.json(r)
                    st.code(json.dumps(r, indent=2, default=str), language="json")
    except Exception as e:
        st.exception(e)

if mode == "ZIP of folders" and run_folders and zip_file:
    if not policy_text:
        policy_text = "Default: approve if amount is reasonable; reject if amount missing or suspicious."
    try:
        svc = get_batch_service()
        doc = svc._doc
        zip_path = save_uploaded_file(zip_file, "batch_zips")
        if not zip_path:
            st.error("Could not save ZIP.")
        else:
            extract_root = Path(doc.upload_dir) / "batch_folders" / zip_path.stem
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_root)
            folder_paths = [extract_root / d.name for d in extract_root.iterdir() if d.is_dir()]
            with st.spinner("Processing folders…"):
                out = svc.process_folders(folder_paths, policy_text=policy_text, client_addresses=client_addresses)
            st.subheader("Results")
            st.metric("Total approved", out["summary"].get("approved", 0))
            st.metric("Total rejected", out["summary"].get("rejected", 0))
            for f in out.get("folders", []):
                with st.expander(f["folder_name"]):
                    st.json(f)
                    st.code(json.dumps(f, indent=2, default=str), language="json")
    except Exception as e:
        st.exception(e)
