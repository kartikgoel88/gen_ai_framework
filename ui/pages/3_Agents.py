"""Agents: invoke ReAct agent (uses framework agents + RAG + MCP)."""

import streamlit as st

st.header("Agents")
st.caption("Run the ReAct agent with RAG and MCP tools (uses framework).")

msg = st.text_area("Message", placeholder="Ask the agent something…")
system_prompt = st.text_input("System prompt (optional)", key="agent_sys")
if st.button("Invoke agent"):
    if not (msg or "").strip():
        st.warning("Enter a message.")
    else:
        try:
            from ui.services import get_agent_client
            agent = get_agent_client()
            with st.spinner("Running agent…"):
                response = agent.invoke(msg.strip(), system_prompt=system_prompt or None)
            st.write(response)
        except Exception as e:
            st.exception(e)
