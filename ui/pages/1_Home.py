"""Home / dashboard."""

import streamlit as st

st.header("Home")
st.markdown("""
- **Batch Expense** — Upload policy and bills (or folders); get approve/reject per bill.
- **Agents** — Run the ReAct agent with RAG and MCP tools.
- **Prompts** — List and run versioned prompts.
- **Graph / RAG** — Run LangGraph RAG or agent.
""")
try:
    from src.framework.config import get_settings
    s = get_settings()
    with st.expander("Framework config (read-only)"):
        st.json({
            "LLM_PROVIDER": s.LLM_PROVIDER,
            "LLM_MODEL": s.LLM_MODEL,
            "UPLOAD_DIR": s.UPLOAD_DIR,
            "VECTOR_STORE": s.VECTOR_STORE,
        })
except Exception as e:
    st.warning(f"Config: {e}")
