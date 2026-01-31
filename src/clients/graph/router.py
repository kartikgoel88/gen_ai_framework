"""Graph API: LangGraph RAG and agent invoke."""

from typing import Any, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.framework.api.deps import get_llm, get_rag, get_mcp_client
from src.framework.config import get_settings_dep, FrameworkSettings
from src.framework.llm.base import LLMClient
from src.framework.rag.base import RAGClient
from src.framework.graph.rag_graph import build_rag_graph
from src.framework.graph.agent_graph import build_agent_graph

router = APIRouter(prefix="/graph", tags=["graph"])


class GraphInvokeRequest(BaseModel):
    query: str
    graph_type: str = "rag"  # rag | agent
    top_k: int = 4


class GraphInvokeResponse(BaseModel):
    response: str
    context: Optional[str] = None
    graph_type: str


@router.post("/invoke", response_model=GraphInvokeResponse)
def graph_invoke(
    body: GraphInvokeRequest,
    llm: LLMClient = Depends(get_llm),
    rag: RAGClient = Depends(get_rag),
    mcp: Any = Depends(get_mcp_client),
    settings: FrameworkSettings = Depends(get_settings_dep),
):
    """Invoke LangGraph: RAG (retrieve -> generate) or ReAct agent. Returns final response."""
    graph_type = (body.graph_type or "rag").lower().strip()
    if graph_type == "rag":
        graph = build_rag_graph(llm=llm, rag=rag, top_k=body.top_k)
        result = graph.invoke({"query": body.query})
        return GraphInvokeResponse(
            response=result.get("response", ""),
            context=result.get("context"),
            graph_type="rag",
        )
    if graph_type == "agent":
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        chat_llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.TEMPERATURE,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        graph = build_agent_graph(llm=chat_llm, rag=rag, mcp_client=mcp)
        result = graph.invoke({"messages": [HumanMessage(content=body.query)]})
        messages = result.get("messages", [])
        last = messages[-1] if messages else None
        response = last.content if hasattr(last, "content") else str(last) if last else ""
        return GraphInvokeResponse(response=response, graph_type="agent")
    return GraphInvokeResponse(response="", graph_type=graph_type)


@router.post("/stream")
def graph_stream(
    body: GraphInvokeRequest,
    llm: LLMClient = Depends(get_llm),
    rag: RAGClient = Depends(get_rag),
):
    """Stream LangGraph RAG output (chunked). Agent graph streaming not implemented here."""
    import json
    if (body.graph_type or "rag").lower().strip() != "rag":
        return {"error": "Only RAG graph supports streaming here. Use /graph/invoke for agent."}
    graph = build_rag_graph(llm=llm, rag=rag, top_k=body.top_k)
    # LangGraph stream: graph.stream({"query": body.query}) yields state updates
    final = graph.invoke({"query": body.query})
    response = final.get("response", "")
    # Emit as single SSE event for simplicity (or chunk by token if LLM supports stream)
    def gen():
        yield f"data: {json.dumps({'chunk': response})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
