"""Prompts: list and run versioned prompts (uses framework prompts store)."""

import streamlit as st

st.header("Prompts")
st.caption("Versioned prompts from framework store.")

store = None
try:
    from src.framework.config import get_settings
    from src.framework.prompts.store import PromptStore
    settings = get_settings()
    base = getattr(settings, "PROMPTS_BASE_PATH", "./data/prompts")
    store = PromptStore(base_path=base)
    names = store.list_names()
except Exception as e:
    names = []
    st.warning(f"Could not load prompt store: {e}")

if store and names:
    name = st.selectbox("Prompt name", names)
    if name:
        versions = store.list_versions(name)
        version = st.selectbox("Version", versions or ["v1"]) if versions else "v1"
        p = store.get(name, version) if versions else store.get(name, "v1")
        if p:
            st.text_area("Body", value=p.body, height=200, disabled=True)
        else:
            st.info("No prompt found.")
else:
    st.info("No prompts in store. Set PROMPTS_BASE_PATH and add prompt files.")
