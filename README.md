# Gen AI Framework

A **modular Gen AI framework** with **separate components** (API, LLM, RAG, agents, chains, graphs, documents, OCR, Docling, MCP, embeddings) and **domain clients** (Onboarding, Admin, Tasks, Agents, Batch, Queue, Prompts, Graph, Evaluation). Built as a reusable framework with pluggable clients, optional task queue (Celery), streaming (SSE), batch inference, versioned prompts, A/B testing, LangGraph, and multiple vector stores.

## Architecture

```
gen_ai_framework/
├── src/
│   ├── framework/              # Reusable framework (separate components)
│   │   ├── api/                # Base API: app factory, dependencies, middleware
│   │   ├── llm/                # LLM abstraction and providers (OpenAI, Grok, Gemini, HuggingFace)
│   │   ├── rag/                # RAG: Chroma (default), Pinecone, Weaviate, Qdrant, pgvector
│   │   ├── embeddings/         # Embeddings: OpenAI, SentenceTransformer
│   │   ├── chains/             # Chains: prompt, RAG, structured, summarization, classification, extraction, pipeline
│   │   ├── graph/              # LangGraph: RAG graph, ReAct agent graph
│   │   ├── agents/             # Agents: ReAct + RAG + MCP tools
│   │   ├── documents/          # Document processor + LangChain loaders
│   │   ├── ocr/                # OCR: EasyOCR for images
│   │   ├── docling/            # Docling: layout-aware doc parsing + OCR
│   │   ├── mcp/                # MCP client: list/call tools (stdio)
│   │   ├── observability/      # Tracing (LLM calls), eval harness
│   │   ├── prompts/            # Versioned prompts, templates, A/B testing
│   │   ├── evaluation/         # Golden datasets, feedback store
│   │   ├── queue/              # Celery tasks: batch RAG, batch bills, agent runs
│   │   └── config.py           # Shared settings
│   │
│   ├── clients/                # Domain-specific clients (use framework)
│   │   ├── onboarding/        # Welcome emails, handbook RAG
│   │   ├── admin/             # Health, config, RAG clear, eval run
│   │   ├── tasks/             # Chat, streaming, batch inference, RAG, chains, OCR, Docling, MCP, embeddings
│   │   ├── agents/            # Invoke agent, list tools, streaming
│   │   ├── batch/             # Cab/meals bills + policy → approve/reject
│   │   ├── queue/             # Enqueue RAG/bills/agent; poll task status
│   │   ├── prompts/           # Versioned prompts, run template, A/B test
│   │   ├── graph/             # LangGraph RAG and agent invoke/stream
│   │   └── evaluation/        # Golden run (regression), feedback, RAG export
│   │
│   └── main.py                # App: mounts all clients
```

- **Framework** = API + LLM + RAG + embeddings + chains + graph + agents + documents + OCR + Docling + MCP + observability + prompts + evaluation + queue as **separate**, injectable components.
- **Clients** = Onboarding, Admin, Tasks, Agents, Batch, Queue, Prompts, Graph, Evaluation; each uses the framework via FastAPI `Depends()`.

## Components (Framework)

| Component        | Role |
|------------------|------|
| **api**          | `create_app()`, CORS, health; deps for LLM, RAG, chains, agent, documents, MCP, etc. |
| **llm**          | Abstract `LLMClient`; providers: OpenAI, Grok (xAI), Gemini, HuggingFace. `stream_invoke` (OpenAI). |
| **rag**          | Abstract `RAGClient`; **ChromaRAG** (default, hybrid BM25, reranker), **PineconeRAG**, **WeaviateRAG**, **QdrantRAG**, **PgvectorRAG**. Chunking strategies, `export_corpus`. |
| **chains**       | `PromptChain`, `RAGChain`, `StructuredChain`, `SummarizationChain`, `ClassificationChain`, `ExtractionChain`, `Pipeline` / `pipeline_from_config`. LangChain LCEL helpers. |
| **graph**        | LangGraph: `build_rag_graph`, `build_agent_graph` (ReAct + RAG + MCP). |
| **agents**       | `LangChainReActAgent`: OpenAI tools agent + RAG + MCP tools. `create_tool_agent()` factory for easy setup. |
| **embeddings**   | OpenAI, SentenceTransformer. |
| **documents**    | DocumentProcessor (PDF/DOCX/Excel/TXT), LangChain loaders. |
| **ocr**          | EasyOCR for images. |
| **docling**      | Layout-aware parsing, OCR for scanned PDFs. |
| **confluence**   | ConfluenceClient: fetch pages by space or ID, HTML→text, ingest into RAG. |
| **mcp**          | MCPClientBridge: stdio server, list/call tools. |
| **observability**| `TracingLLMClient`, `EvalHarness`, `EvalResult`, `evaluate_multiple_models()` for comparing models. |
| **prompts**      | `PromptStore` (versioned files), `TemplateRunner`, `ABTestRunner`. |
| **evaluation**   | `GoldenDatasetRunner`, `FeedbackStore`, RAG export. |
| **queue**        | Celery tasks: batch RAG, batch bills, agent run. |

## Clients & API Summary

| Client        | Prefix           | Purpose |
|---------------|------------------|---------|
| **Onboarding**| `/onboarding`    | Welcome email, handbook ingest, Q&A (RAG+LLM). |
| **Admin**     | `/admin`         | Health, config, RAG clear, **eval run** (dataset path or inline items). |
| **Tasks**     | `/tasks`         | Chat, **chat/stream** (SSE), **batch** (many prompts), RAG ingest/query/retrieve, **chain invoke** (rag, prompt, structured, summarization, classification, extraction, langchain_*), OCR, Docling, LangChain load, MCP, embeddings. |
| **Agents**    | `/agents`        | Invoke agent, **invoke/stream** (SSE), list tools. |
| **Batch**     | `/batch`         | Process bills (flat or folders), policy → approve/reject. |
| **Queue**     | `/tasks/queue`   | **POST** rag / bills / agent → task_id; **GET** status/{task_id}. (Requires Celery + Redis.) |
| **Prompts**   | `/prompts`       | List names/versions, get/put prompt, **run** template, **run/versioned**, **ab-test**. |
| **Graph**     | `/graph`         | **POST** invoke (rag \| agent), **POST** stream (RAG). |
| **Evaluation**| `/evaluation`   | **POST** golden/run (regression), **POST** feedback, **GET** feedback, **POST** feedback/export, **GET** rag/export, **GET** rag/export/download. |

## Setup

From the project root:

```bash
./setup.sh
```

This creates `.venv` (if missing), installs deps (`pip install -e ".[dev]"`), copies `.env.example` to `.env` if needed, and creates `uploads/`, `data/chroma_db/`, `data/prompts/`, `data/feedback/`, `data/golden/`, `output/batch/`. Then edit `.env` and set `OPENAI_API_KEY`.

Manual setup:

```bash
cp .env.example .env
pip install -e ".[dev]"
```

Optional extras:

- **Celery (task queue):** `pip install -e ".[celery]"` and set `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND` (e.g. Redis).
- **Confluence:** `pip install -e ".[confluence]"` and set `CONFLUENCE_BASE_URL`, then Cloud: `CONFLUENCE_EMAIL` + `CONFLUENCE_API_TOKEN`, or Server: `CONFLUENCE_USER` + `CONFLUENCE_PASSWORD`. Use `POST /tasks/rag/ingest/confluence` to ingest pages into RAG.
- **Vector stores:** `pip install -e ".[vectorstore-pinecone]"` (or weaviate, qdrant, pgvector) and set `VECTOR_STORE` + backend env vars.

## Run

**API Server:**
```bash
./run.sh
```

Or:
```bash
uvicorn src.main:app --reload
```

Env: `HOST=0.0.0.0`, `PORT=8000`, `RELOAD=true` (set `RELOAD=false` for production).

- API docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health  

**Streamlit UI:**
```bash
./run.sh ui
```

**Interactive CLI:**
```bash
./run.sh cli-interactive
# or
rag-agent-cli interactive
rag-agent-cli i  # Short alias
```

**CLI Help:**
```bash
./run.sh cli
# or
rag-agent-cli --help
```

**Celery worker** (when using queue):
```bash
celery -A src.framework.queue.app:get_celery_app worker -l info
```

(Ensure `CELERY_BROKER_URL` is set and Redis is running.)

## Example Requests

- **Chat**  
  `POST /tasks/chat` — Body: `{"prompt": "...", "system": "..."}`

- **Chat stream (SSE)**  
  `POST /tasks/chat/stream` — Body: `{"prompt": "...", "system": "..."}` — Events: `data: {"chunk": "..."}`, then `data: [DONE]`

- **Batch inference**  
  `POST /tasks/batch` — Body: `{"prompts": ["...", "..."], "system": "..."}` — Returns `{"responses": ["...", "..."]}`

- **RAG ingest**  
  `POST /tasks/rag/ingest` — Body: `{"text": "...", "metadata": {}}`

- **RAG ingest from Confluence**  
  `POST /tasks/rag/ingest/confluence` — Body: `{"space_key": "DEMO"}` or `{"page_ids": ["123", "456"]}`, optional `limit` (default 100). Set `CONFLUENCE_BASE_URL` and auth (Cloud: `CONFLUENCE_EMAIL` + `CONFLUENCE_API_TOKEN`; Server: `CONFLUENCE_USER` + `CONFLUENCE_PASSWORD`). Optional: `pip install -e ".[confluence]"` for better HTML-to-text.

- **RAG query**  
  `POST /tasks/rag/query` — Body: `{"question": "...", "top_k": 4}`

- **RAG export**  
  `GET /evaluation/rag/export` — JSON corpus  
  `GET /evaluation/rag/export/download` — JSONL download

- **Chain invoke**  
  `POST /tasks/chain/invoke` — Body: `chain_type` = `rag` \| `prompt` \| `structured` \| `summarization` \| `classification` \| `extraction` \| `langchain_prompt` \| `langchain_chat` \| `langchain_rag`, `inputs` = {...}, optional `template`, `labels`, `schema`, `top_k`

- **Agent invoke**  
  `POST /agents/invoke` — Body: `{"message": "...", "system_prompt": "..."}`

- **Agent stream**  
  `POST /agents/invoke/stream` — SSE of final response

- **Graph invoke**  
  `POST /graph/invoke` — Body: `{"query": "...", "graph_type": "rag" \| "agent", "top_k": 4}`

- **Queue (Celery)**  
  `POST /tasks/queue/rag` — Body: `{"question": "...", "top_k": 4}` → `task_id`  
  `POST /tasks/queue/bills` — Body: `{"bill_texts": [...], "policy_text": "...", "file_names": [...]}`  
  `POST /tasks/queue/agent` — Body: `{"message": "...", "system_prompt": "..."}`  
  `GET /tasks/queue/status/{task_id}`

- **Prompts**  
  `GET /prompts/list`, `GET /prompts/{name}/versions`, `GET /prompts/{name}?version=v1`  
  `PUT /prompts/{name}` — Body: `{"body": "...", "version": "v1"}`  
  `POST /prompts/run` — Body: `{"template": "...", "inputs": {}, "structured": false}`  
  `POST /prompts/run/versioned` — Body: `{"name": "...", "version": "v1", "inputs": {}}`  
  `POST /prompts/ab-test` — Body: `{"prompt_a": "...", "prompt_b": "...", "items": [...], "metric": "keyword_match"}`

- **Evaluation**  
  `POST /evaluation/golden/run` — Body: `dataset_path` or `items`, `target` (rag \| batch \| agent), `compare_mode` (exact \| keyword)  
  `POST /evaluation/feedback` — Body: `{"prompt": "...", "response": "...", "feedback": "thumbs_up", "score": 5}`  
  `GET /evaluation/feedback` — Query: `limit`, `session_id`, `has_feedback`  
  `POST /evaluation/feedback/export` — Export for fine-tuning (JSONL)  
  **Multi-model evaluation:** Use `evaluate_multiple_models()` in Python to compare multiple models on the same dataset (see `examples/multi_model_evaluation_example.py`)

- **Admin**  
  `GET /admin/health/components`  
  `GET /admin/config`  
  `POST /admin/rag/clear`  
  `POST /admin/eval/run` — Eval harness (dataset path or inline items)

- **Batch**  
  `POST /batch/process` — Form: policy_file / policy_text, files  
  `POST /batch/process-folders` — Form: folders_zip, policy_file / policy_text  
  CLI: `batch-run --policy policy.pdf --folders ./bills/ -o out.json`

## Testing

**One-command test setup and run** (creates venv if needed, installs dev deps, creates fixture dirs and minimal fixtures, runs pytest):

```bash
./test_setup.sh
```

With options (e.g. verbose, filter by name):

```bash
./test_setup.sh -v
./test_setup.sh -v -k test_rag
./test_setup.sh tests/test_chains.py tests/test_agents.py
```

**Manual:** install dev deps then run pytest:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

- **RAG:** `tests/test_rag.py` — ChromaRAG add/retrieve/export with mock embeddings.
- **Chains:** `tests/test_chains.py` — PromptChain, RAGChain, SummarizationChain, ClassificationChain, ExtractionChain, Pipeline with mock LLM.
- **Graphs:** `tests/test_graphs.py` — RAG graph invoke with mock LLM/RAG.
- **Agents:** `tests/test_agents.py` — Agent tools list and invoke with mock LLM/tools.
- **Batch:** `tests/test_batch_admin_bills.py` — Batch processing from fixtures, writes `output/batch/results.json` (mock LLM).

No API keys required for the above tests (mocks used). For batch test, put `policy.txt` and bill files in `tests/fixtures/data/batch/` (and optional `bills/` subfolder).

## Adding a New Client

1. Add `src/clients/my_client/` with `router.py`.
2. In routes, use `Depends(get_llm)`, `Depends(get_rag)`, etc.
3. In `src/main.py`: `app.include_router(my_client_router)`.
4. In `src/clients/__init__.py`: export the router.

## Agent Architecture & Flow

The framework implements a **ReAct (Reasoning + Acting) agent** pattern using LangChain's agent framework. The agent follows a clear separation of concerns with distinct layers handling different responsibilities.

### Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Layer (Orchestration)                │
│  - LangChainReActAgent: Manages overall flow                │
│  - PromptBuilder: Constructs messages                        │
│  - ToolRegistry: Manages available tools                    │
│  - ToolInterpreter: Parses & executes tool calls            │
│  - EventEmitter: Observability & logging                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              LangChain Agent Graph (ReAct Loop)             │
│  - Manages conversation state                              │
│  - Decides when to use tools                                │
│  - Coordinates LLM calls and tool execution                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Adapter Layer (Message Conversion)              │
│  - LangChainLLMAdapter: Converts LangChain ↔ Provider       │
│  - MessageSerializer: Serializes messages                   │
│  - ToolCallParser: Parses tool calls from text              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    LLM Layer (Provider)                       │
│  - LLMClient: Abstract interface                            │
│  - OpenAIProvider, GeminiProvider, etc.                     │
│  - Native tool calling support (if available)               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Tool Layer (Execution)                    │
│  - RAG Search Tool                                          │
│  - Web Search Tool                                          │
│  - MCP Tools                                                │
│  - Custom Tools                                             │
└─────────────────────────────────────────────────────────────┘
```

### Agent Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│ 1. USER SENDS MESSAGE                                        │
│    "What documents have you ingested?"                       │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 2. PROMPT BUILDER                                            │
│    - SystemMessage: Instructions + tool descriptions          │
│    - HumanMessage: User question                             │
│    - Optional: Chat history prepended                        │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 3. LANGCHAIN AGENT GRAPH                                     │
│    Receives: [SystemMessage, HumanMessage]                    │
│    Manages: ReAct loop (Reasoning + Acting)                  │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 4. LLM ADAPTER CALL                                          │
│    LangChainLLMAdapter._generate(messages)                    │
│    ├─ Serialize messages → Provider format                  │
│    ├─ Call LLMClient.invoke() or invoke_with_tools()        │
│    └─ Return AIMessage with content + optional tool_calls    │
└──────────────────────────────────────────────────────────────┘
                            ↓
                    ┌───────┴───────┐
                    │               │
            Has tool_calls?    No tool_calls?
                    │               │
                    ↓               ↓
┌──────────────────────────┐  ┌──────────────────────────┐
│ 5a. TOOL EXECUTION       │  │ 5b. DIRECT RESPONSE      │
│                          │  │                          │
│ ToolInterpreter:         │  │ Extract content from     │
│ ├─ Parse tool calls      │  │ AIMessage                │
│ ├─ Validate against      │  │                          │
│ │   ToolRegistry         │  │ → Final Answer           │
│ ├─ Execute tools:        │  │                          │
│ │   - rag_search()       │  │                          │
│ │   - web_search()       │  │                          │
│ │   - mcp_tools()        │  │                          │
│ └─ Return ToolMessages   │  │                          │
└──────────────────────────┘  └──────────────────────────┘
                    │
                    ↓
┌──────────────────────────────────────────────────────────────┐
│ 6. TOOL RESULTS ADDED TO CONVERSATION                        │
│    Messages: [SystemMsg, HumanMsg, AIMsg, ToolMsg, ...]     │
└──────────────────────────────────────────────────────────────┘
                    │
                    ↓
            ┌───────┴───────┐
            │               │
    More info needed?   Answer complete?
            │               │
            ↓               ↓
    Loop to step 4    Extract final answer
            │               │
            └───────┬───────┘
                    ↓
┌──────────────────────────────────────────────────────────────┐
│ 7. FINAL RESPONSE                                            │
│    "I have ingested 3 documents: doc1.pdf, doc2.txt, ..."  │
└──────────────────────────────────────────────────────────────┘
```

### Detailed Flow Explanation

#### 1. **Initialization** (`LangChainReActAgent.__init__`)

When an agent is created:

```python
agent = LangChainReActAgent(
    llm=LangChainLLMAdapter(llm_client),
    tools=[rag_search_tool, web_search_tool, ...],
    system_prompt="You are a helpful assistant...",
    verbose=True
)
```

**What happens:**
- **Tool Registry**: All tools are registered in `ToolRegistry` for validation and lookup
- **Tool Interpreter**: Created to handle tool call parsing, validation, and execution
- **Event Emitter**: Set up for observability (logs tool calls, messages, etc.)
- **Prompt Builder**: Initialized with system prompt
- **LangChain Graph**: Created using `create_agent()` with LLM adapter and tools
- **Tool Binding**: Tools are bound to the LLM adapter so it knows what tools are available

#### 2. **Message Processing** (`agent.invoke(message)`)

When a user sends a message:

**Step 2.1: Build Messages**
```python
messages = PromptBuilder.build_messages(user_message)
```
- Combines system prompt, chat history, tool descriptions, and user message
- Returns list of LangChain `BaseMessage` objects

**Step 2.2: Invoke Agent Graph**
```python
result = agent._executor_impl.invoke(messages)
```
- Passes messages to LangChain's agent graph
- The graph manages the ReAct loop internally

#### 3. **ReAct Loop** (Inside LangChain Agent Graph)

The agent graph runs a ReAct loop:

**Step 3.1: LLM Call**
- LangChain calls `LangChainLLMAdapter._generate(messages)`
- Adapter serializes messages to provider format
- Calls underlying `LLMClient.invoke()` or `invoke_with_tools()` if native support
- Returns `AIMessage` with content and optional `tool_calls`

**Step 3.2: Tool Call Detection**
- If `AIMessage.tool_calls` exists (native tool calling):
  - Tool calls are already in LangChain format
  - Skip parsing, go directly to execution
- If no native tool calls but tools are bound:
  - `ToolInterpreter` parses text response for JSON tool calls
  - Validates against tool registry
  - Converts to LangChain format

**Step 3.3: Tool Execution**
- For each tool call:
  - `ToolInterpreter.validate_tool_call()` checks tool exists and arguments are valid
  - `ToolInterpreter.execute_tool_call()` runs the tool
  - Results wrapped in `ToolMessage` objects
  - ToolMessages added to conversation

**Step 3.4: Continue or Finish**
- If more information needed → Loop back to Step 3.1 with tool results
- If answer is complete → Extract final response from last `AIMessage`

#### 4. **Response Extraction**

```python
output = result.get("output", "")
```
- Extracts content from the last message in the conversation
- Returns as string to the user

### Key Components & Responsibilities

| Component | Layer | Responsibility |
|-----------|-------|---------------|
| **LangChainReActAgent** | Agent | Main orchestrator, exposes `invoke()` API, manages lifecycle |
| **PromptBuilder** | Agent | Constructs messages with system prompt, history, and tool descriptions |
| **ToolRegistry** | Agent | Manages available tools, provides tool lookup and validation |
| **ToolInterpreter** | Agent | Parses tool calls from LLM responses, validates, and executes tools |
| **EventEmitter** | Agent | Emits events for observability (tool calls, messages, errors) |
| **LangChain Agent Graph** | Agent | Manages ReAct loop, decides when to use tools, coordinates execution |
| **LangChainLLMAdapter** | Adapter | Converts LangChain messages ↔ Provider format, calls LLM |
| **MessageSerializer** | Adapter | Serializes messages to different formats (string, JSON, native) |
| **ToolCallParser** | Adapter | Parses tool calls from text responses (JSON, structured, native) |
| **LLMClient** | LLM | Abstract interface for LLM providers (OpenAI, Gemini, etc.) |
| **Tools** | Tool | Executable functions (RAG search, web search, MCP tools, custom) |

### Layer Separation Principles

**Agent Layer** ("What do we do with that?")
- Interprets LLM responses
- Decides which tools to use
- Validates and executes tool calls
- Manages conversation flow
- Handles errors and recovery

**Adapter Layer** ("What did the LLM say?")
- Converts message formats
- Calls the LLM provider
- Returns raw LLM responses
- Minimal interpretation (format conversion only)

**LLM Layer** ("Generate text")
- Provider-specific implementations
- Native tool calling support (if available)
- Streaming support
- Error handling

**Tool Layer** ("Execute actions")
- Domain-specific functionality
- RAG retrieval, web search, MCP tools
- Returns structured results

### Tool Execution Flow

When a tool is called, here's the detailed execution path:

```
┌──────────────────────────────────────────────────────────────┐
│ LLM Response                                                  │
│ "I need to search for information about documents"           │
│                                                               │
│ OR Native Tool Call:                                          │
│ AIMessage(tool_calls=[{"name": "rag_search", ...}])         │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Tool Call Detection                                          │
│                                                               │
│ IF Native: tool_calls already in AIMessage                   │
│ IF Text: ToolInterpreter.interpret_response() parses JSON    │
│                                                               │
│ Detected: {"name": "rag_search",                             │
│           "arguments": {"query": "documents", "top_k": 4}}   │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Validation (ToolInterpreter.validate_tool_call())            │
│                                                               │
│ ✓ ToolRegistry.has_tool("rag_search") → True                 │
│ ✓ ToolRegistry.get_tool("rag_search") → RAGSearchTool        │
│ ✓ Arguments match schema:                                   │
│   - query: string (required) ✓                               │
│   - top_k: integer (optional, default 4) ✓                  │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Execution (ToolInterpreter.execute_tool_call())              │
│                                                               │
│ tool = ToolRegistry.get_tool("rag_search")                   │
│ result = tool.invoke({"query": "documents", "top_k": 4})     │
│                                                               │
│ Internally:                                                  │
│   rag_client.retrieve(query="documents", top_k=4)            │
│   → Returns: [                                                │
│       {"content": "doc1.pdf contains...", "metadata": {...}},│
│       {"content": "doc2.txt contains...", "metadata": {...}} │
│     ]                                                         │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Result Wrapping                                               │
│                                                               │
│ ToolMessage(                                                 │
│   content=json.dumps(result),                                │
│   name="rag_search",                                         │
│   tool_call_id="call_rag_search_12345"                       │
│ )                                                             │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Added to Conversation                                         │
│                                                               │
│ Messages: [                                                   │
│   SystemMessage("You are a helpful assistant..."),          │
│   HumanMessage("What documents have you ingested?"),         │
│   AIMessage(tool_calls=[...]),                               │
│   ToolMessage("Found 2 documents: doc1.pdf, doc2.txt")  ← NEW│
│ ]                                                             │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ LLM Processes Tool Results                                    │
│                                                               │
│ LLM sees tool results and generates:                        │
│ "I have ingested 2 documents: doc1.pdf and doc2.txt"       │
└──────────────────────────────────────────────────────────────┘
```

### Example: Complete Flow with Code

Here's a step-by-step example showing what happens internally:

```python
# User code
from src.framework.agents import create_tool_agent
from src.framework.api.deps import get_llm, get_rag

llm = get_llm(settings)
rag = get_rag(settings)

agent = create_tool_agent(
    llm=llm,
    rag_client=rag,
    enable_web_search=True
)

# User sends message
response = agent.invoke("What documents have you ingested?")
```

**Internal Flow:**

```python
# Step 1: PromptBuilder constructs messages
messages = [
    SystemMessage(content="You are a helpful assistant...\nAvailable tools:\n- rag_search: Search documents..."),
    HumanMessage(content="What documents have you ingested?")
]

# Step 2: LangChain agent graph receives messages
# Step 3: LLM is called via adapter
llm_response = adapter._generate(messages)
# Returns: AIMessage(
#     content="",
#     tool_calls=[{"name": "rag_search", "args": {"query": "documents ingested"}, "id": "call_123"}]
# )

# Step 4: Tool interpreter processes tool call
tool_call = ToolCall(name="rag_search", arguments={"query": "documents ingested"})
tool_interpreter.validate_tool_call(tool_call)  # ✓ Valid
tool_result = tool_interpreter.execute_tool_call(tool_call)
# Executes: rag.retrieve("documents ingested", top_k=4)
# Returns: ToolMessage(content='[{"content": "doc1.pdf...", ...}]', name="rag_search")

# Step 5: Tool result added to conversation
messages.append(tool_result)

# Step 6: LLM called again with tool results
llm_response = adapter._generate(messages)
# Returns: AIMessage(content="I have ingested 3 documents: doc1.pdf, doc2.txt, and doc3.docx")

# Step 7: Final response extracted
output = "I have ingested 3 documents: doc1.pdf, doc2.txt, and doc3.docx"
```

**Output:**
```
I have ingested 3 documents: doc1.pdf, doc2.txt, and doc3.docx
```

### Streaming Flow

For streaming responses (`agent.invoke_stream()` or `/agents/invoke/stream`):

```
User Message
    ↓
[Same flow as above]
    ↓
[Streaming Events]
    ├─ AgentEvent(type=THINKING, content="Analyzing...")
    ├─ AgentEvent(type=TOOL_CALL, content="Calling rag_search...")
    ├─ AgentEvent(type=TOOL_RESULT, content="Found 3 documents...")
    └─ AgentEvent(type=MESSAGE_SENT, content="I have ingested...")
    ↓
Final Response (streamed)
```

### Error Handling & Recovery

The agent includes robust error handling:

1. **Tool Call Parsing Errors**: `ToolInterpreter` tries multiple parsing strategies
2. **Tool Validation Errors**: Clear error messages with available tools listed
3. **Tool Execution Errors**: Errors wrapped in ToolMessage, agent can retry or continue
4. **LLM Errors**: Propagated with context, can trigger retries
5. **Recovery**: `ErrorRecoveryManager` attempts automatic recovery for common errors

### Creating Agents

**Simple tool agent (recommended):**
```python
from src.framework.agents import create_tool_agent
from src.framework.api.deps import get_llm, get_rag

llm = get_llm(settings)
rag = get_rag(settings)

agent = create_tool_agent(
    llm=llm,
    rag_client=rag,
    enable_web_search=True,
    system_prompt="You are a helpful assistant.",
)
```

**Using FastAPI dependencies:**
```python
from src.framework.api.deps_agents import get_agent

# In FastAPI route
@app.post("/agents/invoke")
def invoke(message: str, agent: AgentBase = Depends(get_agent)):
    return agent.invoke(message)
```

**Manual setup (advanced):**
```python
from src.framework.agents import LangChainReActAgent, build_framework_tools
from src.framework.adapters import LangChainLLMAdapter

lc_llm = LangChainLLMAdapter(llm_client=llm)
tools = build_framework_tools(rag_client=rag, enable_web_search=True)
agent = LangChainReActAgent(llm=lc_llm, tools=tools)
```

## Examples

See `examples/` directory for:
- **Agent creation:** `create_tool_agent_example.py` — Simple agent setup with tools
- **Multi-model evaluation:** `multi_model_evaluation_example.py` — Compare multiple models on the same dataset
- **Agent features:** `agent_with_tools.py`, `agent_with_memory.py`, `streaming_agent.py`, `planning_agent_example.py`, `reflective_agent_example.py`
- **Multi-agent:** `multi_agent_example.py` — Multiple specialized agents working together
- **Chains:** `chains_example.py` — RAG, prompt, structured chains
- **RAG:** `basic_rag.py` — Document ingestion and querying

## CLI Tools

Command-line tools for common operations (in `src/clients/cli/`):

**Unified RAG + Agent CLI (recommended):**
```bash
# Interactive mode (conversational interface)
rag-agent-cli interactive
rag-agent-cli i  # Short alias
rag-agent-cli i --mode rag  # Start in RAG mode
rag-agent-cli i --no-web-search  # Disable web search
rag-agent-cli i --mcp  # Enable MCP tools
rag-agent-cli i --rag-llm  # Use LLM for RAG queries

# Interactive mode commands (when running):
#   /help     - Show help
#   /mode     - Switch between agent/rag modes
#   /tools    - List available tools
#   /clear    - Clear conversation history
#   /rag      - Quick switch to RAG mode
#   /agent    - Quick switch to agent mode
#   /exit     - Exit interactive mode

# RAG operations
rag-agent-cli rag ingest --file document.txt
rag-agent-cli rag query "What is in the documents?" --top-k 5
rag-agent-cli rag query "Question?" --llm  # Use LLM for answer
rag-agent-cli rag clear
rag-agent-cli rag export --output corpus.jsonl --format jsonl

# Agent operations
rag-agent-cli agent invoke --message "What documents have you ingested?"
rag-agent-cli agent invoke --file question.txt --output answer.txt --web-search
rag-agent-cli agent invoke --message "..." --mcp  # Enable MCP tools
rag-agent-cli agent tools  # List available tools
rag-agent-cli agent tools --json  # JSON output
```

**Separate CLIs (also available):**

**Agent CLI:**
```bash
agent-cli invoke --message "What is RAG?"
agent-cli invoke --file question.txt --output answer.txt
agent-cli tools  # List available tools
agent-cli tools --json  # JSON output
```

**RAG CLI:**
```bash
rag-cli ingest --file document.txt
rag-cli query "What is in the documents?" --top-k 5
rag-cli query "Question?" --llm  # Use LLM for answer
rag-cli clear
rag-cli export --output corpus.jsonl --format jsonl
```

**Chain CLI:**
```bash
chain-cli invoke rag --inputs-json '{"question": "..."}'
chain-cli invoke prompt --inputs-file inputs.json --output result.json
```

Note: For standalone Python scripts, consider using `create_tool_agent()` instead of the CLI's `get_agent()` dependency (see "Creating Agents" section).

## Quick Reference: Agent Flow

**Simple Flow:**
1. User → `agent.invoke(message)` → Agent processes → Returns response
2. Agent uses tools automatically when needed
3. Tools execute and results are fed back to LLM
4. LLM generates final answer

**Key Methods:**
- `agent.invoke(message)` - Synchronous invocation
- `agent.invoke_stream(message)` - Streaming with events
- `agent.get_tools_description()` - List available tools

**Configuration:**
- System prompt: Controls agent behavior and tool usage
- Tools: RAG, web search, MCP tools (configurable)
- Verbose mode: Enables detailed logging of agent steps

## Adding Agent Tools

Extend `framework/agents/tools.py`: add a tool builder and include it in `build_framework_tools()`. The agent gets RAG + MCP tools by default.

**Example: Adding a Custom Tool**

```python
from langchain_core.tools import tool

@tool
def custom_calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)  # In production, use safe eval
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# In build_framework_tools():
tools = [
    rag_search_tool,
    web_search_tool,
    custom_calculator,  # Add your tool
]
```

## Adding LLM / RAG / Vector Store

- **LLM:** Implement `framework.llm.base.LLMClient` and wire in `framework.api.deps.get_llm`.
- **RAG / Vector store:** Implement `framework.rag.base.RAGClient` (e.g. ChromaRAG, PineconeRAG). Set `VECTOR_STORE` and backend env vars; `get_rag` in deps selects the backend.

This keeps the same structure as a **framework** with **separate components** and **clients** for onboarding, admin, tasks, agents, batch, queue, prompts, graph, and evaluation.
