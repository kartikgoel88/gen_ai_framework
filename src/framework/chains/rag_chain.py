"""RAG chain: retrieve context from RAG client, then generate answer with LLM."""

from typing import Any

from ..llm.base import LLMClient
from ..rag.base import RAGClient

from .base import Chain

DEFAULT_RAG_PROMPT = """Use the following context to answer the question. If the context does not contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""


class RAGChain(Chain):
    """Chain that retrieves context from RAG and generates an answer with the LLM."""

    def __init__(
        self,
        llm: LLMClient,
        rag: RAGClient,
        prompt_template: str = DEFAULT_RAG_PROMPT,
        top_k: int = 4,
    ):
        """
        Args:
            llm: LLM client for generating the answer.
            rag: RAG client for retrieval (retrieve).
            prompt_template: Template with {context} and {question}.
            top_k: Number of chunks to retrieve.
        """
        self._llm = llm
        self._rag = rag
        self._prompt_template = prompt_template
        self._top_k = top_k

    def invoke(self, inputs: dict[str, Any], **kwargs: Any) -> str:
        """Run RAG: retrieve with query, format prompt, invoke LLM. Input key: 'query' or 'question'."""
        query = inputs.get("query") or inputs.get("question") or ""
        top_k = kwargs.get("top_k", self._top_k)
        chunks = self._rag.retrieve(query, top_k=top_k, **kwargs)
        context = "\n\n".join(c.get("content", "") for c in chunks)
        prompt = self._prompt_template.format(context=context, question=query)
        return self._llm.invoke(prompt, **kwargs)
