"""Graph / RAG: run LangGraph RAG or agent (uses framework graph + RAG)."""

import streamlit as st

st.header("Graph / RAG")
st.caption("LangGraph RAG (retrieve → generate) or agent. Uses framework LLM + RAG.")

query = st.text_input("Query", placeholder="Ask a question…")
graph_type = st.radio("Graph type", ["rag", "agent"], horizontal=True)
top_k = st.number_input("RAG top_k", min_value=1, max_value=20, value=4)
if st.button("Invoke"):
    if not (query or "").strip():
        st.warning("Enter a query.")
    else:
        try:
            from src.framework.config import get_settings
            from src.framework.api.deps import get_llm, get_rag, get_mcp_client
            from src.framework.graph.rag_graph import build_rag_graph
            from src.framework.graph.agent_graph import build_agent_graph
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage

            settings = get_settings()
            llm = get_llm(settings)
            rag = get_rag(settings)
            if graph_type == "rag":
                graph = build_rag_graph(llm=llm, rag=rag, top_k=top_k)
                result = graph.invoke({"query": query.strip()})
                st.write(result.get("response", ""))
                if result.get("context"):
                    with st.expander("Context"):
                        st.text(result["context"])
            else:
                mcp = get_mcp_client(settings)
                chat_llm = ChatOpenAI(
                    model=settings.LLM_MODEL,
                    temperature=settings.TEMPERATURE,
                    openai_api_key=settings.OPENAI_API_KEY,
                )
                graph = build_agent_graph(llm=chat_llm, rag=rag, mcp_client=mcp)
                result = graph.invoke({"messages": [HumanMessage(content=query.strip())]})
                messages = result.get("messages", [])
                last = messages[-1] if messages else None
                text = last.content if last and hasattr(last, "content") else str(result)
                st.write(text)
        except Exception as e:
            st.exception(e)
