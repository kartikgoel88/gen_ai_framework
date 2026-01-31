"""LangGraph RAG workflow: retrieve -> generate."""

from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage

from ..rag.base import RAGClient


class RAGGraphState(TypedDict):
    """State for RAG graph: query, retrieved context, final response."""

    query: str
    context: str
    response: str


def build_rag_graph(
    llm: Any,
    rag: RAGClient,
    top_k: int = 4,
    system_prompt: Optional[str] = None,
):
    """
    Build a LangGraph RAG workflow: retrieve -> generate.
    llm: LangChain chat model (e.g. ChatOpenAI) for compatibility with LangGraph.
    rag: Framework RAGClient for retrieval.
    Returns compiled graph.
    """
    system_prompt = system_prompt or "Use the context below to answer the question. If the context does not contain enough information, say so."

    def retrieve_node(state: RAGGraphState) -> dict[str, Any]:
        query = state.get("query") or (state.get("messages") and state["messages"][-1].content) or ""
        chunks = rag.retrieve(query, top_k=top_k)
        context = "\n\n".join(c.get("content", "") for c in chunks)
        return {"context": context, "query": query}

    def generate_node(state: RAGGraphState) -> dict[str, Any]:
        context = state.get("context", "")
        query = state.get("query", "")
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        # If llm is framework LLMClient, we need to invoke it; if LangChain model, invoke with messages
        if hasattr(llm, "invoke") and callable(llm.invoke):
            response = llm.invoke(prompt)
        else:
            msg = llm.invoke([HumanMessage(content=prompt)])
            response = msg.content if hasattr(msg, "content") else str(msg)
        return {"response": response}

    workflow = StateGraph(RAGGraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


def _default_rag_state() -> type:
    """Minimal state for RAG (query, context, response only)."""
    class State(TypedDict):
        query: str
        context: str
        response: str
    return State
