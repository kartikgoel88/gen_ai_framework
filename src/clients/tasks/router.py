"""Tasks client API routes: generic LLM, RAG, OCR, Docling, LangChain docs, MCP, embeddings, chains, streaming, batch."""

import json
from typing import Optional, Any, List

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.framework.api.deps import (
    get_llm,
    get_rag,
    get_rag_chain,
    get_document_processor,
    get_langchain_loader,
    get_ocr_processor,
    get_docling_processor,
    get_confluence_client,
    get_mcp_client,
    get_embeddings,
)
from src.framework.confluence.client import ConfluenceClient
from src.framework.llm.base import LLMClient
from src.framework.rag.base import RAGClient
from src.framework.chains import (
    Chain,
    PromptChain,
    RAGChain,
    StructuredChain,
    SummarizationChain,
    ClassificationChain,
    ExtractionChain,
    build_langchain_prompt_chain,
    build_langchain_chat_prompt_chain,
    build_langchain_rag_chain,
)
from src.framework.chains.summarization_chain import DEFAULT_SUMMARY_PROMPT
from src.framework.documents.processor import DocumentProcessor
from src.framework.documents.langchain_loader import LangChainDocProcessor
from src.framework.ocr.processor import OcrProcessor
from src.framework.docling.processor import DoclingProcessor
from src.framework.mcp.client import MCPClientBridge
from src.framework.embeddings.base import EmbeddingsProvider

router = APIRouter(prefix="/tasks", tags=["tasks"])


class ChatRequest(BaseModel):
    prompt: str
    system: Optional[str] = None


class ChatResponse(BaseModel):
    response: str


class RAGIngestRequest(BaseModel):
    text: str
    metadata: Optional[dict[str, Any]] = None


class RAGQueryRequest(BaseModel):
    question: str
    top_k: int = 4


@router.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    llm: LLMClient = Depends(get_llm),
):
    """Generic chat completion using the framework LLM."""
    prompt = req.prompt
    if req.system:
        prompt = f"{req.system}\n\n{prompt}"
    response = llm.invoke(prompt)
    return ChatResponse(response=response)


def _sse_stream(generator):
    """Yield SSE-formatted chunks: data: {json}\n\n."""
    for chunk in generator:
        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/chat/stream")
def chat_stream(
    req: ChatRequest,
    llm: LLMClient = Depends(get_llm),
):
    """Stream chat completion as Server-Sent Events. Each event: data: {\"chunk\": \"...\"}; final event: data: [DONE]."""
    prompt = req.prompt
    if req.system:
        prompt = f"{req.system}\n\n{prompt}"
    try:
        stream = llm.stream_invoke(prompt)
    except NotImplementedError:
        return {"error": "Streaming not supported by configured LLM provider"}
    return StreamingResponse(
        _sse_stream(stream),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class BatchInferenceRequest(BaseModel):
    """Many prompts in one request for high throughput."""

    prompts: List[str]
    system: Optional[str] = None


class BatchInferenceResponse(BaseModel):
    responses: List[str]


@router.post("/batch", response_model=BatchInferenceResponse)
def batch_inference(
    body: BatchInferenceRequest,
    llm: LLMClient = Depends(get_llm),
):
    """Batch inference: run many prompts in one request. Returns list of responses in same order."""
    prefix = f"{body.system}\n\n" if body.system else ""
    responses = []
    for p in body.prompts:
        full = f"{prefix}{p}" if prefix else p
        responses.append(llm.invoke(full))
    return BatchInferenceResponse(responses=responses)


@router.post("/extract")
async def extract_from_document(
    file: UploadFile = File(...),
    doc_processor: DocumentProcessor = Depends(get_document_processor),
):
    """Extract text from an uploaded document (PDF, DOCX, etc.)."""
    content = await file.read()
    path = doc_processor.save_upload(content, file.filename or "upload")
    result = doc_processor.extract(path)
    return result.to_dict()


@router.post("/rag/ingest")
def rag_ingest(
    body: RAGIngestRequest,
    rag: RAGClient = Depends(get_rag),
):
    """Ingest raw text into the RAG store."""
    rag.add_documents([body.text], metadatas=[body.metadata] if body.metadata else None)
    return {"ok": True}


class ConfluenceIngestRequest(BaseModel):
    """Ingest Confluence pages into RAG. Provide space_key and/or page_ids."""

    space_key: Optional[str] = None
    page_ids: Optional[List[str]] = None
    limit: int = 100


@router.post("/rag/ingest/confluence")
def rag_ingest_confluence(
    body: ConfluenceIngestRequest,
    rag: RAGClient = Depends(get_rag),
    confluence: ConfluenceClient | None = Depends(get_confluence_client),
):
    """Ingest Confluence pages into the RAG store. Set CONFLUENCE_BASE_URL (and auth) in .env."""
    if confluence is None:
        raise HTTPException(
            status_code=503,
            detail="Confluence is not configured. Set CONFLUENCE_BASE_URL and CONFLUENCE_EMAIL/CONFLUENCE_API_TOKEN (Cloud) or CONFLUENCE_USER/CONFLUENCE_PASSWORD (Server).",
        )
    if not body.space_key and not body.page_ids:
        raise HTTPException(
            status_code=400,
            detail="Provide space_key and/or page_ids.",
        )
    pages = confluence.fetch_pages_for_ingest(
        space_key=body.space_key,
        page_ids=body.page_ids,
        limit=body.limit,
    )
    if not pages:
        return {"ok": True, "ingested": 0, "message": "No pages found or no content extracted."}
    texts = [p[0] for p in pages]
    metadatas = [p[1] for p in pages]
    rag.add_documents(texts, metadatas=metadatas)
    return {"ok": True, "ingested": len(texts)}


@router.post("/rag/query")
def rag_query(
    body: RAGQueryRequest,
    rag: RAGClient = Depends(get_rag),
    llm: LLMClient = Depends(get_llm),
):
    """Query the RAG store and optionally get an LLM-generated answer."""
    answer = rag.query(body.question, llm_client=llm, top_k=body.top_k)
    return {"question": body.question, "answer": answer}


# --- Chains ---


class ChainInvokeRequest(BaseModel):
    """Invoke a chain with inputs. chain_type: rag | prompt | structured | summarization | classification | extraction | langchain_*."""

    chain_type: str = "rag"
    inputs: dict[str, Any] = {}
    template: Optional[str] = None
    system: Optional[str] = None
    human_template: Optional[str] = None
    top_k: Optional[int] = None
    labels: Optional[List[str]] = None  # For classification chain
    output_schema: Optional[str] = Field(None, alias="schema")  # For extraction chain (e.g. "title, author, date")


@router.post("/chain/invoke")
def chain_invoke(
    body: ChainInvokeRequest,
    llm: LLMClient = Depends(get_llm),
    rag_chain: RAGChain = Depends(get_rag_chain),
    rag: RAGClient = Depends(get_rag),
):
    """Invoke a chain: rag, prompt, structured, or LangChain LCEL (langchain_prompt, langchain_chat, langchain_rag)."""
    chain_type = (body.chain_type or "rag").lower().strip()
    inputs = body.inputs or {}
    if chain_type == "rag":
        return {"output": rag_chain.invoke(inputs)}
    if chain_type == "prompt":
        template = body.template or "{prompt}"
        chain = PromptChain(llm=llm, template=template)
        return {"output": chain.invoke(inputs)}
    if chain_type == "structured":
        template = body.template or "{prompt}"
        chain = StructuredChain(llm=llm, template=template)
        return {"output": chain.invoke(inputs)}
    # LangChain LCEL chains (use framework LLM/RAG under the hood)
    if chain_type == "langchain_prompt":
        template = body.template or "{prompt}"
        chain = build_langchain_prompt_chain(llm=llm, template=template)
        return {"output": chain.invoke(inputs)}
    if chain_type == "langchain_chat":
        chain = build_langchain_chat_prompt_chain(
            llm=llm,
            system=body.system,
            human_template=body.human_template or "{input}",
        )
        return {"output": chain.invoke(inputs)}
    if chain_type == "langchain_rag":
        top_k = body.top_k if body.top_k is not None else 4
        chain = build_langchain_rag_chain(llm=llm, rag=rag, top_k=top_k)
        return {"output": chain.invoke(inputs)}
    if chain_type == "summarization":
        template = body.template or DEFAULT_SUMMARY_PROMPT
        chain = SummarizationChain(llm=llm, prompt_template=template)
        return {"output": chain.invoke(inputs)}
    if chain_type == "classification":
        labels = body.labels or inputs.get("labels") or ["positive", "negative", "neutral"]
        chain = ClassificationChain(llm=llm, labels=labels)
        return {"output": chain.invoke(inputs)}
    if chain_type == "extraction":
        schema = body.output_schema or inputs.get("schema") or "key facts and entities"
        chain = ExtractionChain(llm=llm, schema=schema)
        return {"output": chain.invoke(inputs)}
    return {
        "error": (
            f"Unknown chain_type: {body.chain_type}. "
            "Use rag, prompt, structured, summarization, classification, extraction, langchain_prompt, langchain_chat, or langchain_rag."
        )
    }


@router.get("/rag/retrieve")
def rag_retrieve(
    q: str,
    top_k: int = 4,
    rag: RAGClient = Depends(get_rag),
):
    """Retrieve relevant chunks only (no LLM answer)."""
    chunks = rag.retrieve(q, top_k=top_k)
    return {"query": q, "chunks": chunks}


# --- OCR ---


@router.post("/ocr")
async def ocr_image(
    file: UploadFile = File(...),
    ocr: OcrProcessor = Depends(get_ocr_processor),
):
    """Run OCR on an uploaded image (PNG, JPG, etc.)."""
    content = await file.read()
    result = ocr.extract_from_bytes(content)
    return result.to_dict()


@router.post("/ocr/file")
async def ocr_image_file(
    file: UploadFile = File(...),
    doc_processor: DocumentProcessor = Depends(get_document_processor),
    ocr: OcrProcessor = Depends(get_ocr_processor),
):
    """Run OCR on an uploaded image file (saved to disk first)."""
    path = doc_processor.save_upload(await file.read(), file.filename or "image.png")
    result = ocr.extract(path)
    return result.to_dict()


# --- Docling ---


@router.post("/docling")
async def docling_extract(
    file: UploadFile = File(...),
    format: str = "markdown",
    doc_processor: DocumentProcessor = Depends(get_document_processor),
    docling: DoclingProcessor = Depends(get_docling_processor),
):
    """Extract text from document using Docling (layout-aware, OCR for scanned PDFs)."""
    path = doc_processor.save_upload(await file.read(), file.filename or "doc.pdf")
    result = docling.extract(path, export_format=format)
    return result.to_dict()


# --- LangChain doc loaders ---


@router.post("/langchain-load")
async def langchain_load(
    file: UploadFile = File(...),
    doc_processor: DocumentProcessor = Depends(get_document_processor),
    loader: LangChainDocProcessor = Depends(get_langchain_loader),
):
    """Load document using LangChain loaders (PyPDF, Docx2txt, Text, CSV)."""
    path = doc_processor.save_upload(await file.read(), file.filename or "upload")
    result = loader.load_as_result(path)
    return result.to_dict()


@router.get("/langchain-load/chunks")
async def langchain_load_chunks(
    file_path: str,
    loader: LangChainDocProcessor = Depends(get_langchain_loader),
):
    """Load document and return list of LangChain Document chunks (by path)."""
    docs = loader.load(file_path)
    return {
        "chunks": [
            {"page_content": d.page_content, "metadata": d.metadata}
            for d in docs
        ],
    }


# --- MCP ---


class MCPCallRequest(BaseModel):
    name: str
    arguments: Optional[dict[str, Any]] = None


@router.get("/mcp/tools")
def mcp_list_tools(
    mcp: MCPClientBridge = Depends(get_mcp_client),
):
    """List tools exposed by the configured MCP server."""
    tools = mcp.list_tools()
    return {"tools": tools}


@router.post("/mcp/call")
def mcp_call_tool(
    body: MCPCallRequest,
    mcp: MCPClientBridge = Depends(get_mcp_client),
):
    """Call an MCP tool by name with optional arguments."""
    return mcp.call_tool(body.name, body.arguments)


# --- Embeddings ---


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedQueryRequest(BaseModel):
    text: str


@router.post("/embeddings/embed")
def embeddings_embed(
    body: EmbedRequest,
    embeddings: EmbeddingsProvider = Depends(get_embeddings),
):
    """Embed a list of texts using the configured embeddings model."""
    vectors = embeddings.embed_documents(body.texts)
    return {"embeddings": vectors, "dim": len(vectors[0]) if vectors else 0}


@router.post("/embeddings/embed-query")
def embeddings_embed_query(
    body: EmbedQueryRequest,
    embeddings: EmbeddingsProvider = Depends(get_embeddings),
):
    """Embed a single query string."""
    vector = embeddings.embed_query(body.text)
    return {"embedding": vector, "dim": len(vector)}
