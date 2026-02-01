"""Gen AI Framework — Streamlit UI (separate from clients and framework)."""

import sys
from pathlib import Path

# Ensure project root is on path so src.* and ui.* resolve
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

st.set_page_config(
    page_title="Gen AI Framework",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Gen AI Framework")
st.markdown("Use the **sidebar** to open: **Batch Expense**, **Agents**, **Prompts**, **Graph/RAG**.")
st.info("This UI is a separate app that uses `src.clients` and `src.framework` as libraries.")
