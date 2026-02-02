"""Agent and chain dependencies for FastAPI.

This module provides dependency injection functions for agents and chains,
including RAG chains and ReAct agents with tool support.
"""

from typing import Annotated

from fastapi import Depends

from ..agents.base import AgentBase
from ..agents.langchain_agent import LangChainReActAgent
from ..agents.tools import build_framework_tools
from ..chains.rag_chain import RAGChain
from ..llm.base import LLMClient
from ..rag.base import RAGClient
from ..config import get_settings_dep, FrameworkSettings
from .deps_llm import get_llm
from .deps_rag import get_rag
from .deps_integrations import get_mcp_client


def get_rag_chain(
    llm: LLMClient = Depends(get_llm),
    rag: RAGClient = Depends(get_rag),
) -> RAGChain:
    """Dependency that returns the RAG chain.
    
    Combines RAG retrieval with LLM generation for question-answering
    over documents.
    
    Args:
        llm: LLM client (injected via Depends)
        rag: RAG client (injected via Depends)
        
    Returns:
        RAGChain instance with default top_k=4
        
    Example:
        ```python
        @app.post("/rag/query")
        def query(question: str, chain: RAGChain = Depends(get_rag_chain)):
            return chain.invoke({"question": question})
        ```
    """
    return RAGChain(llm=llm, rag=rag, top_k=4)


def get_agent(
    settings: Annotated[FrameworkSettings, Depends(get_settings_dep)],
    rag: Annotated[RAGClient, Depends(get_rag)],
    mcp: Annotated[object, Depends(get_mcp_client)],
) -> AgentBase:
    """Dependency that returns the configured agent.
    
    Creates a ReAct agent with access to:
    - RAG tool: Search ingested documents
    - Web search tool: Search the internet (including LinkedIn)
    - MCP tools: Additional tools from MCP servers
    
    Args:
        settings: Framework settings (injected via FastAPI Depends)
        rag: RAG client for document search (injected)
        mcp: MCP client bridge for additional tools (injected)
        
    Returns:
        AgentBase instance (LangChainReActAgent)
        
    Example:
        ```python
        @app.post("/agents/invoke")
        def invoke_agent(message: str, agent: AgentBase = Depends(get_agent)):
            return agent.invoke(message)
        ```
    """
    from langchain_openai import ChatOpenAI

    tools = build_framework_tools(rag_client=rag, mcp_client=mcp, enable_web_search=True)
    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=settings.TEMPERATURE,
        openai_api_key=settings.OPENAI_API_KEY,
    )
    return LangChainReActAgent(llm=llm, tools=tools, verbose=settings.DEBUG)
