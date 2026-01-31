"""Agents client API routes."""

import json
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.framework.api.deps import get_agent
from src.framework.agents.base import AgentBase

router = APIRouter(prefix="/agents", tags=["agents"])


class AgentInvokeRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = None


class AgentInvokeResponse(BaseModel):
    response: str


@router.post("/invoke", response_model=AgentInvokeResponse)
def agent_invoke(
    req: AgentInvokeRequest,
    agent: AgentBase = Depends(get_agent),
):
    """Run the agent with a message. Agent can use RAG and MCP tools when configured."""
    response = agent.invoke(
        req.message,
        system_prompt=req.system_prompt,
    )
    return AgentInvokeResponse(response=response)


def _agent_sse_stream(text: str, chunk_size: int = 50):
    """Yield SSE events for agent response (chunked for progressive display)."""
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/invoke/stream")
def agent_invoke_stream(
    req: AgentInvokeRequest,
    agent: AgentBase = Depends(get_agent),
):
    """Run the agent and stream the final response as Server-Sent Events (chunked)."""
    response = agent.invoke(req.message, system_prompt=req.system_prompt)
    return StreamingResponse(
        _agent_sse_stream(response),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/tools")
def agent_tools(
    agent: AgentBase = Depends(get_agent),
):
    """List tools available to the agent (RAG search, MCP tools, etc.)."""
    tools = agent.get_tools_description()
    return {"tools": tools}
