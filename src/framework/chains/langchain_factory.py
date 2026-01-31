"""Build LangChain LCEL chains from framework LLMClient (and optional RAG)."""

from typing import Any, List, Optional

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..llm.base import LLMClient
from ..rag.base import RAGClient

from .langchain_adapter import LangChainLLMAdapter
from .langchain_chain import LangChainChain
from .rag_chain import DEFAULT_RAG_PROMPT


def build_langchain_prompt_chain(
    llm: LLMClient,
    template: str,
    input_variables: Optional[List[str]] = None,
) -> LangChainChain:
    """
    Build a LangChain LCEL chain: PromptTemplate | LLM (adapter) | StrOutputParser.
    Returns a framework Chain that accepts inputs dict and returns a string.
    """
    adapter = LangChainLLMAdapter(llm_client=llm)
    if input_variables is None:
        import re
        input_variables = list(re.findall(r"\{(\w+)\}", template))
    prompt = PromptTemplate(template=template, input_variables=input_variables)
    runnable = prompt | adapter | StrOutputParser()
    return LangChainChain(runnable=runnable)


def build_langchain_chat_prompt_chain(
    llm: LLMClient,
    system: Optional[str] = None,
    human_template: str = "{input}",
) -> LangChainChain:
    """
    Build a LangChain LCEL chain with chat-style prompt (system + human).
    Inputs should include 'input' (and optionally 'context', etc.).
    """
    adapter = LangChainLLMAdapter(llm_client=llm)
    messages = []
    if system:
        messages.append(("system", system))
    messages.append(("human", human_template))
    prompt = ChatPromptTemplate.from_messages(messages)
    runnable = prompt | adapter | StrOutputParser()
    return LangChainChain(runnable=runnable)


def build_langchain_rag_chain(
    llm: LLMClient,
    rag: RAGClient,
    prompt_template: str = DEFAULT_RAG_PROMPT,
    top_k: int = 4,
) -> LangChainChain:
    """
    Build a LangChain-style RAG chain: retrieve context, then prompt | LLM | StrOutputParser.
    The runnable expects input with key 'query' or 'question'.
    """
    adapter = LangChainLLMAdapter(llm_client=llm)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    def retrieve_and_invoke(inputs: dict[str, Any]) -> dict[str, Any]:
        query = inputs.get("query") or inputs.get("question") or ""
        chunks = rag.retrieve(query, top_k=top_k)
        context = "\n\n".join(c.get("content", "") for c in chunks)
        return {"context": context, "question": query}

    # LCEL: we need a runnable that takes dict with query -> outputs dict with context, question -> then prompt | llm | parser
    # LangChain's RunnablePassthrough.assign can do this. Simpler: use a custom RunnableLambda that does retrieve then invokes prompt | adapter | parser.
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

    def run_rag(inputs: dict[str, Any]) -> str:
        query = inputs.get("query") or inputs.get("question") or ""
        chunks = rag.retrieve(query, top_k=top_k)
        context = "\n\n".join(c.get("content", "") for c in chunks)
        formatted = prompt.format(context=context, question=query)
        return adapter.invoke(formatted) if hasattr(adapter, "invoke") else adapter.invoke([__import__("langchain_core.messages", fromlist=["HumanMessage"]).HumanMessage(content=formatted)])

    # adapter is BaseChatModel, so it expects messages; we need to pass a prompt string as HumanMessage
    from langchain_core.messages import HumanMessage

    def run_rag_invoke(inputs: dict[str, Any]) -> str:
        query = inputs.get("query") or inputs.get("question") or ""
        chunks = rag.retrieve(query, top_k=top_k)
        context = "\n\n".join(c.get("content", "") for c in chunks)
        formatted = prompt.format(context=context, question=query)
        result = adapter.invoke([HumanMessage(content=formatted)])
        return result.content if hasattr(result, "content") else str(result)

    runnable = RunnableLambda(run_rag_invoke)
    return LangChainChain(runnable=runnable)
