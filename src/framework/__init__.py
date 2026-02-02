"""Gen AI Framework: Reusable components for building AI applications.

This framework provides modular, pluggable components for:
- **LLM Integration**: Multiple providers (OpenAI, Grok, Gemini, HuggingFace)
- **RAG Pipeline**: Vector stores (Chroma, Pinecone, Weaviate, Qdrant, pgvector)
- **Document Processing**: PDF, DOCX, TXT extraction with OCR support
- **Agents & Tools**: ReAct agents with RAG, web search, and MCP tools
- **Chains**: Prompt, RAG, structured, summarization, classification chains
- **LangGraph**: Graph-based workflows for complex AI pipelines
- **Observability**: LLM tracing, evaluation harness, feedback collection

Architecture:
    The framework is organized into separate, injectable components:
    
    - **api/**: FastAPI dependencies and app factory
    - **llm/**: LLM abstraction with provider implementations
    - **rag/**: RAG clients for various vector stores
    - **embeddings/**: Embedding providers (OpenAI, SentenceTransformers)
    - **documents/**: Document extraction and processing
    - **ocr/**: OCR processing for images
    - **agents/**: Agent implementations with tool support
    - **chains/**: Chain implementations for common tasks
    - **graph/**: LangGraph workflow builders
    - **observability/**: Tracing and evaluation tools
    - **prompts/**: Versioned prompt management
    - **evaluation/**: Golden datasets and feedback collection

Quick Start:
    ```python
    from src.framework.config import get_settings
    from src.framework.api.deps import get_llm, get_rag
    
    settings = get_settings()
    llm = get_llm(settings)
    rag = get_rag(settings)
    
    # Use components
    response = llm.invoke("Hello!")
    rag.add_documents(["Document text..."])
    results = rag.retrieve("query", top_k=5)
    ```

For more information, see:
    - README.md: Full documentation and examples
    - REFACTORING_RECOMMENDATIONS.md: Code structure improvements
"""

from .config import get_settings, get_settings_dep

__all__ = ["get_settings", "get_settings_dep"]
