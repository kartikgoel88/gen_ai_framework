"""Batch Expense: process bills against policy (uses clients.batch + framework)."""

import json
import zipfile
from pathlib import Path

import streamlit as st

from ui.services import get_batch_service, get_doc_processor, save_uploaded_file


st.header("Batch Expense")
st.caption("Process cab/meals bills against admin policy. Uses framework LLM + document processor.")

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
                extracted = (result.text or "").strip()
                st.session_state["policy_text"] = result.text or ""
                if extracted:
                    st.success("Policy extracted. Now upload bill files below and click **Process bills** to see approval/rejection results.")
                else:
                    reason = result.error or "Unknown reason."
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
client_addresses = {}
if client_addresses_raw and client_addresses_raw.strip():
    try:
        client_addresses = json.loads(client_addresses_raw)
        if not isinstance(client_addresses, dict):
            client_addresses = {}
    except Exception:
        st.warning("Invalid JSON for client addresses; ignoring.")

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
    except Exception as e:
        st.exception(e)
