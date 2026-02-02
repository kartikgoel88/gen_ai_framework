"""Shared configuration for the framework."""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrameworkSettings(BaseSettings):
    """Framework-wide settings."""

    OPENAI_API_KEY: Optional[str] = None
    LLM_MODEL: str = "gpt-4-turbo-preview"
    TEMPERATURE: float = 0.7

    # LLM provider: openai | grok | gemini | huggingface
    LLM_PROVIDER: str = "openai"
    # Grok (xAI): OpenAI-compatible API at api.x.ai
    XAI_API_KEY: Optional[str] = None
    # Gemini (Google)
    GOOGLE_API_KEY: Optional[str] = None
    # Hugging Face (e.g. Qwen): HUGGINGFACEHUB_API_TOKEN or HF_TOKEN
    HUGGINGFACE_API_KEY: Optional[str] = None

    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Embeddings provider: openai | sentence_transformers
    EMBEDDINGS_PROVIDER: str = "openai"
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"

    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    SECRET_KEY: str = "change-me"
    DEBUG: bool = True

    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE: int = 10485760

    # MCP (optional): command and args for stdio server, e.g. ["python", "mcp_server.py"]
    MCP_COMMAND: Optional[str] = None
    MCP_ARGS: Optional[str] = None  # JSON array string, e.g. '["server.py"]'

    # Observability: LLM tracing (log prompt/response/latency)
    ENABLE_LLM_TRACING: bool = False
    TRACING_LOG_LEVEL: str = "INFO"  # DEBUG | INFO | WARNING

    # RAG: chunking strategy (recursive_character | sentence)
    CHUNKING_STRATEGY: str = "recursive_character"
    # RAG: hybrid search (vector + BM25)
    RAG_HYBRID_SEARCH: bool = False
    # RAG: rerank top candidates (0 = disabled, else take top_k * this factor then rerank to top_k)
    RAG_RERANK_TOP_N: int = 0  # e.g. 2 = retrieve 2*top_k, rerank to top_k

    # Task queue (Celery): leave empty to disable queue endpoints
    CELERY_BROKER_URL: Optional[str] = None  # e.g. redis://localhost:6379/0
    CELERY_RESULT_BACKEND: Optional[str] = None  # e.g. redis://localhost:6379/0

    # Prompts: versioned prompts base path (files or DB); file-based = directory path
    PROMPTS_BASE_PATH: str = "./data/prompts"
    # A/B test: default metric for comparison (exact_match | keyword_match | latency)
    PROMPT_AB_METRIC: str = "keyword_match"

    # Evaluation: golden datasets and feedback
    GOLDEN_DATASETS_PATH: str = "./data/golden"
    FEEDBACK_STORE_PATH: str = "./data/feedback/feedback.jsonl"

    # Vector store: chroma | pinecone | weaviate | qdrant | pgvector (default chroma)
    VECTOR_STORE: str = "chroma"
    # Pinecone
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: Optional[str] = None
    PINECONE_ENV: Optional[str] = None
    # Weaviate
    WEAVIATE_URL: Optional[str] = None
    WEAVIATE_INDEX_NAME: str = "Chunk"
    # Qdrant
    QDRANT_URL: Optional[str] = None
    QDRANT_COLLECTION: str = "rag"
    # pgvector (Postgres connection string)
    PGVECTOR_CONNECTION_STRING: Optional[str] = None
    PGVECTOR_TABLE: str = "rag_embeddings"

    # Confluence (optional): ingest pages into RAG
    CONFLUENCE_BASE_URL: Optional[str] = None  # e.g. https://your-site.atlassian.net/wiki
    CONFLUENCE_EMAIL: Optional[str] = None  # Cloud: email for API token auth
    CONFLUENCE_API_TOKEN: Optional[str] = None  # Cloud: API token
    CONFLUENCE_USER: Optional[str] = None  # Server/DC: username for basic auth
    CONFLUENCE_PASSWORD: Optional[str] = None  # Server/DC: password

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


@lru_cache
def get_settings() -> FrameworkSettings:
    return FrameworkSettings()


def get_settings_dep() -> FrameworkSettings:
    """FastAPI dependency that returns settings."""
    return get_settings()
