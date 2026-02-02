"""Nested configuration models for better organization.

This module provides nested Pydantic models for configuration, grouping
related settings together for better organization and validation.

The nested structure is available alongside the flat structure for
backward compatibility.
"""

from typing import Optional
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    """LLM provider configuration."""
    provider: str = Field(default="openai", description="LLM provider: openai | grok | gemini | huggingface")
    model: str = Field(default="gpt-4-turbo-preview", description="Model name/identifier")
    temperature: float = Field(default=0.7, description="Sampling temperature (0.0-2.0)")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    xai_api_key: Optional[str] = Field(default=None, description="xAI (Grok) API key")
    google_api_key: Optional[str] = Field(default=None, description="Google (Gemini) API key")
    huggingface_api_key: Optional[str] = Field(default=None, description="HuggingFace API key")


class EmbeddingsSettings(BaseModel):
    """Embeddings provider configuration."""
    provider: str = Field(default="openai", description="Embeddings provider: openai | sentence_transformers")
    model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    sentence_transformer_model: str = Field(default="all-MiniLM-L6-v2", description="SentenceTransformer model name")


class RAGSettings(BaseModel):
    """RAG (Retrieval-Augmented Generation) configuration."""
    vector_store: str = Field(default="chroma", description="Vector store: chroma | pinecone | weaviate | qdrant | pgvector")
    chunk_size: int = Field(default=1000, description="Text chunk size")
    chunk_overlap: int = Field(default=200, description="Chunk overlap size")
    chunking_strategy: str = Field(default="recursive_character", description="Chunking strategy: recursive_character | sentence")
    hybrid_search: bool = Field(default=False, description="Enable hybrid search (vector + BM25)")
    rerank_top_n: int = Field(default=0, description="Reranking factor (0 = disabled)")
    persist_dir: str = Field(default="./data/chroma_db", description="ChromaDB persistence directory")
    
    # Vector store specific settings
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: Optional[str] = None
    pinecone_env: Optional[str] = None
    weaviate_url: Optional[str] = None
    weaviate_index_name: str = "Chunk"
    qdrant_url: Optional[str] = None
    qdrant_collection: str = "rag"
    pgvector_connection_string: Optional[str] = None
    pgvector_table: str = "rag_embeddings"


class ObservabilitySettings(BaseModel):
    """Observability and tracing configuration."""
    enable_llm_tracing: bool = Field(default=False, description="Enable LLM call tracing")
    tracing_log_level: str = Field(default="INFO", description="Tracing log level: DEBUG | INFO | WARNING")


class QueueSettings(BaseModel):
    """Task queue (Celery) configuration."""
    broker_url: Optional[str] = Field(default=None, description="Celery broker URL (e.g., redis://localhost:6379/0)")
    result_backend: Optional[str] = Field(default=None, description="Celery result backend URL")


class PromptsSettings(BaseModel):
    """Versioned prompts configuration."""
    base_path: str = Field(default="./data/prompts", description="Base path for versioned prompts")
    ab_metric: str = Field(default="keyword_match", description="Default A/B test metric: exact_match | keyword_match | latency")


class EvaluationSettings(BaseModel):
    """Evaluation and feedback configuration."""
    golden_datasets_path: str = Field(default="./data/golden", description="Path to golden datasets")
    feedback_store_path: str = Field(default="./data/feedback/feedback.jsonl", description="Path to feedback store")


class ConfluenceSettings(BaseModel):
    """Confluence integration configuration."""
    base_url: Optional[str] = Field(default=None, description="Confluence base URL")
    email: Optional[str] = Field(default=None, description="Confluence email (Cloud)")
    api_token: Optional[str] = Field(default=None, description="Confluence API token (Cloud)")
    username: Optional[str] = Field(default=None, description="Confluence username (Server/DC)")
    password: Optional[str] = Field(default=None, description="Confluence password (Server/DC)")


class MCPSettings(BaseModel):
    """MCP (Model Context Protocol) configuration."""
    command: Optional[str] = Field(default=None, description="MCP server command")
    args: Optional[str] = Field(default=None, description="MCP server args (JSON array string)")


class FrameworkSettingsNested(BaseSettings):
    """Framework settings with nested organization.
    
    This class provides a nested structure for better organization while
    maintaining backward compatibility with the flat structure.
    """
    
    # Core settings
    secret_key: str = "change-me"
    debug: bool = True
    upload_dir: str = "./uploads"
    max_upload_size: int = 10485760
    
    # Nested settings groups
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embeddings: EmbeddingsSettings = Field(default_factory=EmbeddingsSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    queue: QueueSettings = Field(default_factory=QueueSettings)
    prompts: PromptsSettings = Field(default_factory=PromptsSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    confluence: ConfluenceSettings = Field(default_factory=ConfluenceSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)
    
    def __init__(self, **kwargs):
        """Initialize with backward compatibility for flat env vars.
        
        Maps flat environment variables to nested structure for compatibility.
        """
        # Map flat env vars to nested structure
        flat_to_nested = {
            # LLM
            "LLM_PROVIDER": ("llm", "provider"),
            "LLM_MODEL": ("llm", "model"),
            "TEMPERATURE": ("llm", "temperature"),
            "OPENAI_API_KEY": ("llm", "openai_api_key"),
            "XAI_API_KEY": ("llm", "xai_api_key"),
            "GOOGLE_API_KEY": ("llm", "google_api_key"),
            "HUGGINGFACE_API_KEY": ("llm", "huggingface_api_key"),
            # Embeddings
            "EMBEDDINGS_PROVIDER": ("embeddings", "provider"),
            "EMBEDDING_MODEL": ("embeddings", "model"),
            "SENTENCE_TRANSFORMER_MODEL": ("embeddings", "sentence_transformer_model"),
            # RAG
            "VECTOR_STORE": ("rag", "vector_store"),
            "CHUNK_SIZE": ("rag", "chunk_size"),
            "CHUNK_OVERLAP": ("rag", "chunk_overlap"),
            "CHUNKING_STRATEGY": ("rag", "chunking_strategy"),
            "RAG_HYBRID_SEARCH": ("rag", "hybrid_search"),
            "RAG_RERANK_TOP_N": ("rag", "rerank_top_n"),
            "CHROMA_PERSIST_DIR": ("rag", "persist_dir"),
            "PINECONE_API_KEY": ("rag", "pinecone_api_key"),
            "PINECONE_INDEX_NAME": ("rag", "pinecone_index_name"),
            "PINECONE_ENV": ("rag", "pinecone_env"),
            "WEAVIATE_URL": ("rag", "weaviate_url"),
            "WEAVIATE_INDEX_NAME": ("rag", "weaviate_index_name"),
            "QDRANT_URL": ("rag", "qdrant_url"),
            "QDRANT_COLLECTION": ("rag", "qdrant_collection"),
            "PGVECTOR_CONNECTION_STRING": ("rag", "pgvector_connection_string"),
            "PGVECTOR_TABLE": ("rag", "pgvector_table"),
            # Observability
            "ENABLE_LLM_TRACING": ("observability", "enable_llm_tracing"),
            "TRACING_LOG_LEVEL": ("observability", "tracing_log_level"),
            # Queue
            "CELERY_BROKER_URL": ("queue", "broker_url"),
            "CELERY_RESULT_BACKEND": ("queue", "result_backend"),
            # Prompts
            "PROMPTS_BASE_PATH": ("prompts", "base_path"),
            "PROMPT_AB_METRIC": ("prompts", "ab_metric"),
            # Evaluation
            "GOLDEN_DATASETS_PATH": ("evaluation", "golden_datasets_path"),
            "FEEDBACK_STORE_PATH": ("evaluation", "feedback_store_path"),
            # Confluence
            "CONFLUENCE_BASE_URL": ("confluence", "base_url"),
            "CONFLUENCE_EMAIL": ("confluence", "email"),
            "CONFLUENCE_API_TOKEN": ("confluence", "api_token"),
            "CONFLUENCE_USER": ("confluence", "username"),
            "CONFLUENCE_PASSWORD": ("confluence", "password"),
            # MCP
            "MCP_COMMAND": ("mcp", "command"),
            "MCP_ARGS": ("mcp", "args"),
        }
        
        # Process flat env vars
        nested_kwargs = {}
        for key, value in kwargs.items():
            if key in flat_to_nested:
                group, attr = flat_to_nested[key]
                if group not in nested_kwargs:
                    nested_kwargs[group] = {}
                nested_kwargs[group][attr] = value
            else:
                nested_kwargs[key] = value
        
        super().__init__(**nested_kwargs)
    
    # Backward compatibility properties
    @property
    def LLM_PROVIDER(self) -> str:
        return self.llm.provider
    
    @property
    def LLM_MODEL(self) -> str:
        return self.llm.model
    
    @property
    def TEMPERATURE(self) -> float:
        return self.llm.temperature
    
    @property
    def OPENAI_API_KEY(self) -> Optional[str]:
        return self.llm.openai_api_key
    
    @property
    def XAI_API_KEY(self) -> Optional[str]:
        return self.llm.xai_api_key
    
    @property
    def GOOGLE_API_KEY(self) -> Optional[str]:
        return self.llm.google_api_key
    
    @property
    def HUGGINGFACE_API_KEY(self) -> Optional[str]:
        return self.llm.huggingface_api_key
    
    @property
    def EMBEDDINGS_PROVIDER(self) -> str:
        return self.embeddings.provider
    
    @property
    def EMBEDDING_MODEL(self) -> str:
        return self.embeddings.model
    
    @property
    def SENTENCE_TRANSFORMER_MODEL(self) -> str:
        return self.embeddings.sentence_transformer_model
    
    @property
    def VECTOR_STORE(self) -> str:
        return self.rag.vector_store
    
    @property
    def CHUNK_SIZE(self) -> int:
        return self.rag.chunk_size
    
    @property
    def CHUNK_OVERLAP(self) -> int:
        return self.rag.chunk_overlap
    
    @property
    def CHUNKING_STRATEGY(self) -> str:
        return self.rag.chunking_strategy
    
    @property
    def RAG_HYBRID_SEARCH(self) -> bool:
        return self.rag.hybrid_search
    
    @property
    def RAG_RERANK_TOP_N(self) -> int:
        return self.rag.rerank_top_n
    
    @property
    def CHROMA_PERSIST_DIR(self) -> str:
        return self.rag.persist_dir
    
    @property
    def PINECONE_API_KEY(self) -> Optional[str]:
        return self.rag.pinecone_api_key
    
    @property
    def PINECONE_INDEX_NAME(self) -> Optional[str]:
        return self.rag.pinecone_index_name
    
    @property
    def PINECONE_ENV(self) -> Optional[str]:
        return self.rag.pinecone_env
    
    @property
    def WEAVIATE_URL(self) -> Optional[str]:
        return self.rag.weaviate_url
    
    @property
    def WEAVIATE_INDEX_NAME(self) -> str:
        return self.rag.weaviate_index_name
    
    @property
    def QDRANT_URL(self) -> Optional[str]:
        return self.rag.qdrant_url
    
    @property
    def QDRANT_COLLECTION(self) -> str:
        return self.rag.qdrant_collection
    
    @property
    def PGVECTOR_CONNECTION_STRING(self) -> Optional[str]:
        return self.rag.pgvector_connection_string
    
    @property
    def PGVECTOR_TABLE(self) -> str:
        return self.rag.pgvector_table
    
    @property
    def ENABLE_LLM_TRACING(self) -> bool:
        return self.observability.enable_llm_tracing
    
    @property
    def TRACING_LOG_LEVEL(self) -> str:
        return self.observability.tracing_log_level
    
    @property
    def CELERY_BROKER_URL(self) -> Optional[str]:
        return self.queue.broker_url
    
    @property
    def CELERY_RESULT_BACKEND(self) -> Optional[str]:
        return self.queue.result_backend
    
    @property
    def PROMPTS_BASE_PATH(self) -> str:
        return self.prompts.base_path
    
    @property
    def PROMPT_AB_METRIC(self) -> str:
        return self.prompts.ab_metric
    
    @property
    def GOLDEN_DATASETS_PATH(self) -> str:
        return self.evaluation.golden_datasets_path
    
    @property
    def FEEDBACK_STORE_PATH(self) -> str:
        return self.evaluation.feedback_store_path
    
    @property
    def CONFLUENCE_BASE_URL(self) -> Optional[str]:
        return self.confluence.base_url
    
    @property
    def CONFLUENCE_EMAIL(self) -> Optional[str]:
        return self.confluence.email
    
    @property
    def CONFLUENCE_API_TOKEN(self) -> Optional[str]:
        return self.confluence.api_token
    
    @property
    def CONFLUENCE_USER(self) -> Optional[str]:
        return self.confluence.username
    
    @property
    def CONFLUENCE_PASSWORD(self) -> Optional[str]:
        return self.confluence.password
    
    @property
    def MCP_COMMAND(self) -> Optional[str]:
        return self.mcp.command
    
    @property
    def MCP_ARGS(self) -> Optional[str]:
        return self.mcp.args
