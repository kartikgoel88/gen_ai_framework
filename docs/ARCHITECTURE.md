# Framework Architecture

## Overview

The Gen AI Framework is designed as a modular, pluggable system with clear separation between:
- **Framework Components**: Reusable, injectable components
- **Domain Clients**: Application-specific logic that uses the framework

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│  (FastAPI Routes, Streamlit UI, CLI Tools)                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Dependency Injection                      │
│              (FastAPI Depends, Service Layer)               │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   LLM Layer │ │   RAG Layer  │ │  Documents   │
│             │ │              │ │              │
│ - OpenAI    │ │ - Chroma     │ │ - PDF        │
│ - Grok      │ │ - Pinecone   │ │ - DOCX       │
│ - Gemini    │ │ - Weaviate   │ │ - OCR        │
│ - HuggingFace│ │ - Qdrant     │ │ - Docling    │
└──────────────┘ │ - pgvector   │ └──────────────┘
                 └──────────────┘
        │              │              │
        └──────────────┼──────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Configuration Layer                       │
│              (Settings, Environment Variables)              │
└─────────────────────────────────────────────────────────────┘
```

## Component Layers

### 1. Core Framework Components

#### LLM Layer (`framework/llm/`)
- **Base**: `LLMClient` abstract interface
- **Providers**: OpenAI, Grok, Gemini, HuggingFace
- **Registry**: Provider registration and factory pattern
- **Features**: Streaming, structured output, chat interface

#### RAG Layer (`framework/rag/`)
- **Base**: `RAGClient` abstract interface
- **Backends**: Chroma, Pinecone, Weaviate, Qdrant, pgvector
- **Features**: Hybrid search, reranking, chunking strategies
- **Registry**: Vector store registration pattern

#### Embeddings Layer (`framework/embeddings/`)
- **Base**: `EmbeddingsProvider` abstract interface
- **Providers**: OpenAI, SentenceTransformer
- **Usage**: Used by RAG layer for vector embeddings

#### Documents Layer (`framework/documents/`)
- **Processor**: Document extraction (PDF, DOCX, TXT, Excel)
- **OCR**: Image-based OCR (EasyOCR)
- **Docling**: Layout-aware parsing
- **LangChain**: LangChain document loaders

#### Agents Layer (`framework/agents/`)
- **Base**: `AgentBase` abstract interface
- **Implementation**: LangChain ReAct agent
- **Tools**: RAG tool, web search, MCP tools
- **Features**: Tool selection, reasoning, streaming

#### Chains Layer (`framework/chains/`)
- **Base**: `Chain` abstract interface
- **Types**: Prompt, RAG, Structured, Summarization, Classification, Extraction
- **Pipeline**: Multi-step chain composition
- **LangChain**: LangChain LCEL integration

#### Graph Layer (`framework/graph/`)
- **RAG Graph**: LangGraph-based RAG workflow
- **Agent Graph**: LangGraph-based agent workflow
- **Features**: State management, conditional flows

### 2. Dependency Injection (`framework/api/`)

The dependency injection layer provides FastAPI dependencies for all components:

- **deps_llm.py**: LLM provider dependencies
- **deps_rag.py**: RAG backend dependencies
- **deps_embeddings.py**: Embeddings dependencies
- **deps_documents.py**: Document processing dependencies
- **deps_agents.py**: Agent and chain dependencies
- **deps_integrations.py**: External integrations (Confluence, MCP)

### 3. Configuration (`framework/config.py`)

- **Flat Structure**: Original flat configuration (backward compatible)
- **Nested Structure**: New nested configuration (`config_nested.py`)
- **Environment Variables**: Loaded from `.env` file
- **Validation**: Pydantic-based validation

### 4. Domain Clients (`clients/`)

Domain-specific applications that use the framework:

- **onboarding/**: Welcome emails, handbook RAG
- **admin/**: Health checks, configuration, RAG management
- **tasks/**: Chat, batch inference, RAG operations, chains
- **agents/**: Agent invocation and streaming
- **batch/**: Batch document processing
- **queue/**: Celery task queue integration
- **prompts/**: Versioned prompt management
- **graph/**: LangGraph workflows
- **evaluation/**: Golden datasets, feedback collection

## Design Patterns

### 1. Provider Registry Pattern

Eliminates if/else chains by using self-registering providers:

```python
@LLMProviderRegistry.register("openai")
def create_openai(**kwargs):
    return OpenAILLMProvider(**kwargs)

llm = LLMProviderRegistry.create(provider="openai", **kwargs)
```

### 2. Dependency Injection

FastAPI's `Depends()` provides clean dependency injection:

```python
@app.post("/chat")
def chat(llm: LLMClient = Depends(get_llm)):
    return llm.invoke("Hello!")
```

### 3. Abstract Base Classes

All components implement abstract interfaces:

```python
class LLMClient(ABC):
    @abstractmethod
    def invoke(self, prompt: str) -> str:
        ...
```

### 4. Factory Pattern

Cached factory functions for component creation:

```python
@lru_cache
def _get_llm_cached(provider, api_key, model, temperature):
    return LLMProviderRegistry.create(...)
```

## Data Flow

### RAG Query Flow

```
User Query
    │
    ▼
RAG Client
    │
    ├─► Embeddings Provider (vectorize query)
    │
    ├─► Vector Store (similarity search)
    │
    ├─► Reranker (optional, improve results)
    │
    └─► LLM Client (generate answer with context)
    │
    ▼
Final Answer
```

### Agent Flow

```
User Message
    │
    ▼
Agent
    │
    ├─► Tool Selection (RAG | Web Search | MCP)
    │
    ├─► Tool Execution
    │
    └─► LLM Reasoning
    │
    ▼
Response
```

## Extension Points

### Adding a New LLM Provider

1. Implement `LLMClient` interface
2. Register with `LLMProviderRegistry`
3. Add configuration settings (if needed)

### Adding a New Vector Store

1. Implement `RAGClient` interface
2. Register with `RAGProviderRegistry`
3. Add configuration settings

### Adding a New Agent Tool

1. Create LangChain `BaseTool` implementation
2. Add to `build_framework_tools()` in `agents/tools.py`

### Adding a New Chain Type

1. Implement `Chain` interface
2. Add to `chains/__init__.py`
3. Create dependency function if needed

## Error Handling

Custom exception hierarchy:

- `FrameworkError`: Base exception
- `ProviderNotFoundError`: Provider not registered
- `ConfigurationError`: Invalid configuration
- `APIKeyError`: Missing API key
- `VectorStoreError`: Vector store operation failed

## Testing Strategy

- **Unit Tests**: Test individual components
- **Integration Tests**: Test component interactions
- **Mock Providers**: Use mocks for external APIs
- **Test Fixtures**: Reusable test data and configurations

## Performance Considerations

- **Caching**: LRU cache for expensive operations (LLM, embeddings)
- **Lazy Loading**: Optional dependencies loaded only when used
- **Streaming**: Support for streaming responses
- **Batch Processing**: Batch inference for multiple prompts

## Security Considerations

- **API Keys**: Stored in environment variables, never in code
- **Input Validation**: Pydantic validation for all inputs
- **Rate Limiting**: Can be added at FastAPI level
- **Access Control**: Can be added via FastAPI dependencies

## Future Enhancements

- **Async Support**: Full async/await support
- **Distributed Tracing**: OpenTelemetry integration
- **Metrics**: Prometheus metrics export
- **Caching Layer**: Redis-based caching
- **Multi-tenancy**: Support for multiple tenants
