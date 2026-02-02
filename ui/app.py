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

st.title("🤖 Gen AI Framework")
st.markdown("""
Welcome! Navigate using the sidebar:
- **Home**: Feature overview and quick start
- **Gen AI Learning**: End-to-end flows and interactive tutorials
- **Batch Expense**: Complete application example (bills reimbursement)
""")
st.info("💡 **New to the framework?** Start with **Home** to see all features, then explore **Gen AI Learning** for complete workflows.")
