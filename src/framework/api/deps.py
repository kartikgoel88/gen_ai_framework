"""FastAPI dependencies for framework components."""

import json
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from ..llm.base import LLMClient
from ..llm.openai_provider import OpenAILLMProvider
from ..llm.grok_provider import GrokLLMProvider
from ..llm.gemini_provider import GeminiLLMProvider
from ..llm.huggingface_provider import HuggingFaceLLMProvider
from ..rag.base import RAGClient
from ..rag.chroma_rag import ChromaRAG
from ..rag.reranker import CrossEncoderReranker
from ..embeddings.base import EmbeddingsProvider
from ..embeddings.openai_provider import OpenAIEmbeddingsProvider
from ..embeddings.sentence_transformer_provider import SentenceTransformerEmbeddingsProvider
from ..documents.processor import DocumentProcessor
from ..documents.pdf_ocr_processor import PdfOcrProcessor
from ..documents.langchain_loader import LangChainDocProcessor
from ..ocr.processor import OcrProcessor
from ..docling.processor import DoclingProcessor
from ..confluence.client import ConfluenceClient
from ..mcp.client import MCPClientBridge
from ..agents.base import AgentBase
from ..agents.langchain_agent import LangChainReActAgent
from ..agents.tools import build_framework_tools
from ..chains.rag_chain import RAGChain
from ..config import get_settings_dep, FrameworkSettings
from ..observability.tracing import TracingLLMClient


def _create_llm_provider(
    provider: str,
    api_key: str | None,
    model: str,
    temperature: float,
) -> LLMClient:
    """Create LLM client for the given provider (openai | grok | gemini | huggingface)."""
    p = (provider or "openai").lower().strip()
    if p == "grok":
        if not api_key:
            raise ValueError("XAI_API_KEY is required for Grok. Set LLM_PROVIDER=grok and XAI_API_KEY.")
        return GrokLLMProvider(api_key=api_key, model=model, temperature=temperature)
    if p == "gemini":
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for Gemini. Set LLM_PROVIDER=gemini and GOOGLE_API_KEY.")
        return GeminiLLMProvider(api_key=api_key, model=model, temperature=temperature)
    if p == "huggingface":
        if not api_key:
            raise ValueError(
                "HUGGINGFACE_API_KEY is required for Hugging Face. Set LLM_PROVIDER=huggingface and HUGGINGFACE_API_KEY."
            )
        return HuggingFaceLLMProvider(api_key=api_key, model=model, temperature=temperature)
    # openai (default)
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for OpenAI. Set OPENAI_API_KEY.")
    return OpenAILLMProvider(api_key=api_key, model=model, temperature=temperature)


@lru_cache
def _get_llm_cached(
    provider: str,
    api_key: str | None,
    model: str,
    temperature: float,
) -> LLMClient:
    return _create_llm_provider(provider, api_key, model, temperature)


def get_llm(
    settings: Annotated[FrameworkSettings, Depends(get_settings_dep)],
) -> LLMClient:
    """Dependency that returns the configured LLM client (openai | grok | gemini | huggingface). Optionally wrapped with TracingLLMClient if ENABLE_LLM_TRACING."""
    import logging

    provider = (settings.LLM_PROVIDER or "openai").lower().strip()
    if provider == "grok":
        api_key = settings.XAI_API_KEY
    elif provider == "gemini":
        api_key = settings.GOOGLE_API_KEY
    elif provider == "huggingface":
        api_key = settings.HUGGINGFACE_API_KEY
    else:
        api_key = settings.OPENAI_API_KEY
    llm = _get_llm_cached(
        provider=provider,
        api_key=api_key,
        model=settings.LLM_MODEL,
        temperature=settings.TEMPERATURE,
    )
    if getattr(settings, "ENABLE_LLM_TRACING", False):
        level = getattr(logging, (settings.TRACING_LOG_LEVEL or "INFO").upper(), logging.INFO)
        llm = TracingLLMClient(llm, log_level=level)
    return llm


@lru_cache
def _get_embeddings_provider(
    provider: str,
    openai_model: str,
    openai_api_key: str | None,
    st_model: str,
) -> EmbeddingsProvider:
    if provider == "sentence_transformers":
        return SentenceTransformerEmbeddingsProvider(model_name=st_model)
    return OpenAIEmbeddingsProvider(model=openai_model, openai_api_key=openai_api_key)


def get_embeddings(
    settings: Annotated[FrameworkSettings, Depends(get_settings_dep)],
) -> EmbeddingsProvider:
    """Dependency that returns the configured embeddings provider."""
    return _get_embeddings_provider(
        provider=settings.EMBEDDINGS_PROVIDER,
        openai_model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
        st_model=settings.SENTENCE_TRANSFORMER_MODEL,
    )


def _get_reranker_if_enabled(rerank_top_n: int):
    """Return a CrossEncoderReranker if rerank_top_n > 0, else None."""
    if rerank_top_n <= 0:
        return None
    try:
        return CrossEncoderReranker()
    except Exception:
        return None


@lru_cache
def _get_rag_client(
    persist_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    chunking_strategy: str,
    use_hybrid: bool,
    rerank_top_n: int,
    provider: str,
    openai_model: str,
    openai_api_key: str | None,
    st_model: str,
) -> RAGClient:
    emb = _get_embeddings_provider(provider, openai_model, openai_api_key, st_model)
    reranker = _get_reranker_if_enabled(rerank_top_n) if rerank_top_n > 0 else None
    return ChromaRAG(
        persist_directory=persist_dir,
        embeddings=emb,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunking_strategy=chunking_strategy,
        use_hybrid=use_hybrid,
        reranker=reranker,
        rerank_top_n=rerank_top_n,
    )


def get_rag(
    settings: Annotated[FrameworkSettings, Depends(get_settings_dep)],
) -> RAGClient:
    """Dependency that returns the configured RAG client. Supports chroma (default), pinecone, weaviate, qdrant, pgvector."""
    store = (getattr(settings, "VECTOR_STORE", None) or "chroma").lower().strip()
    if store == "pinecone" and getattr(settings, "PINECONE_API_KEY", None) and getattr(settings, "PINECONE_INDEX_NAME", None):
        from ..rag.pinecone_rag import PineconeRAG
        emb = _get_embeddings_provider(
            settings.EMBEDDINGS_PROVIDER,
            settings.EMBEDDING_MODEL,
            settings.OPENAI_API_KEY,
            settings.SENTENCE_TRANSFORMER_MODEL,
        )
        return PineconeRAG(
            embeddings=emb,
            index_name=settings.PINECONE_INDEX_NAME,
            api_key=settings.PINECONE_API_KEY,
            environment=getattr(settings, "PINECONE_ENV", None),
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
    if store == "weaviate" and getattr(settings, "WEAVIATE_URL", None):
        from ..rag.weaviate_rag import WeaviateRAG
        emb = _get_embeddings_provider(
            settings.EMBEDDINGS_PROVIDER,
            settings.EMBEDDING_MODEL,
            settings.OPENAI_API_KEY,
            settings.SENTENCE_TRANSFORMER_MODEL,
        )
        return WeaviateRAG(
            embeddings=emb,
            url=settings.WEAVIATE_URL,
            index_name=getattr(settings, "WEAVIATE_INDEX_NAME", "Chunk"),
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
    if store == "qdrant" and getattr(settings, "QDRANT_URL", None):
        from ..rag.qdrant_rag import QdrantRAG
        emb = _get_embeddings_provider(
            settings.EMBEDDINGS_PROVIDER,
            settings.EMBEDDING_MODEL,
            settings.OPENAI_API_KEY,
            settings.SENTENCE_TRANSFORMER_MODEL,
        )
        return QdrantRAG(
            embeddings=emb,
            url=settings.QDRANT_URL,
            collection_name=getattr(settings, "QDRANT_COLLECTION", "rag"),
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
    if store == "pgvector" and getattr(settings, "PGVECTOR_CONNECTION_STRING", None):
        from ..rag.pgvector_rag import PgvectorRAG
        emb = _get_embeddings_provider(
            settings.EMBEDDINGS_PROVIDER,
            settings.EMBEDDING_MODEL,
            settings.OPENAI_API_KEY,
            settings.SENTENCE_TRANSFORMER_MODEL,
        )
        return PgvectorRAG(
            embeddings=emb,
            connection_string=settings.PGVECTOR_CONNECTION_STRING,
            table_name=getattr(settings, "PGVECTOR_TABLE", "rag_embeddings"),
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
    return _get_rag_client(
        persist_dir=settings.CHROMA_PERSIST_DIR,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        chunking_strategy=getattr(settings, "CHUNKING_STRATEGY", "recursive_character"),
        use_hybrid=getattr(settings, "RAG_HYBRID_SEARCH", False),
        rerank_top_n=getattr(settings, "RAG_RERANK_TOP_N", 0),
        provider=settings.EMBEDDINGS_PROVIDER,
        openai_model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
        st_model=settings.SENTENCE_TRANSFORMER_MODEL,
    )


def get_pdf_ocr_processor() -> PdfOcrProcessor:
    """Dependency that returns the PDF processor (PyMuPDF + pytesseract OCR fallback)."""
    return PdfOcrProcessor(dpi=300, min_text_len=10)


def get_document_processor(
    settings: Annotated[FrameworkSettings, Depends(get_settings_dep)],
    pdf_ocr_processor: PdfOcrProcessor = Depends(get_pdf_ocr_processor),
) -> DocumentProcessor:
    """Dependency that returns the document processor (PDFs use PyMuPDF + pytesseract when pdf_ocr_processor is injected)."""
    return DocumentProcessor(upload_dir=settings.UPLOAD_DIR, pdf_processor=pdf_ocr_processor)


def get_langchain_loader() -> LangChainDocProcessor:
    """Dependency that returns the LangChain document loader."""
    return LangChainDocProcessor()


def get_ocr_processor() -> OcrProcessor:
    """Dependency that returns the OCR processor."""
    return OcrProcessor()


def get_rag_chain(
    llm: LLMClient = Depends(get_llm),
    rag: RAGClient = Depends(get_rag),
) -> RAGChain:
    """Dependency that returns the RAG chain (retrieve + LLM with default prompt)."""
    return RAGChain(llm=llm, rag=rag, top_k=4)


def get_docling_processor() -> DoclingProcessor:
    """Dependency that returns the Docling processor."""
    return DoclingProcessor()


def get_confluence_client(
    settings: Annotated[FrameworkSettings, Depends(get_settings_dep)],
) -> ConfluenceClient | None:
    """Dependency that returns the Confluence client when CONFLUENCE_BASE_URL is set; otherwise None."""
    base_url = getattr(settings, "CONFLUENCE_BASE_URL", None) or ""
    if not base_url.strip():
        return None
    email = getattr(settings, "CONFLUENCE_EMAIL", None)
    api_token = getattr(settings, "CONFLUENCE_API_TOKEN", None)
    username = getattr(settings, "CONFLUENCE_USER", None)
    password = getattr(settings, "CONFLUENCE_PASSWORD", None)
    return ConfluenceClient(
        base_url=base_url,
        email=email,
        api_token=api_token,
        username=username,
        password=password,
    )


def get_mcp_client(
    settings: Annotated[FrameworkSettings, Depends(get_settings_dep)],
) -> MCPClientBridge:
    """Dependency that returns the MCP client bridge (if configured)."""
    command = settings.MCP_COMMAND or "python"
    args = []
    if settings.MCP_ARGS:
        try:
            args = json.loads(settings.MCP_ARGS)
        except Exception:
            pass
    return MCPClientBridge(command=command, args=args, env={})


def get_agent(
    settings: Annotated[FrameworkSettings, Depends(get_settings_dep)],
    rag: Annotated[RAGClient, Depends(get_rag)],
    mcp: Annotated[MCPClientBridge, Depends(get_mcp_client)],
) -> AgentBase:
    """Dependency that returns the configured agent (ReAct + RAG + MCP tools)."""
    from langchain_openai import ChatOpenAI

    tools = build_framework_tools(rag_client=rag, mcp_client=mcp)
    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=settings.TEMPERATURE,
        openai_api_key=settings.OPENAI_API_KEY,
    )
    return LangChainReActAgent(llm=llm, tools=tools, verbose=settings.DEBUG)
