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
| **agents**       | `LangChainReActAgent`: OpenAI tools agent + RAG + MCP tools. |
| **embeddings**   | OpenAI, SentenceTransformer. |
| **documents**    | DocumentProcessor (PDF/DOCX/Excel/TXT), LangChain loaders. |
| **ocr**          | EasyOCR for images. |
| **docling**      | Layout-aware parsing, OCR for scanned PDFs. |
| **mcp**          | MCPClientBridge: stdio server, list/call tools. |
| **observability**| `TracingLLMClient`, `EvalHarness`, `EvalResult`. |
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
- **Vector stores:** `pip install -e ".[vectorstore-pinecone]"` (or weaviate, qdrant, pgvector) and set `VECTOR_STORE` + backend env vars.

## Run

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

## Adding Agent Tools

Extend `framework/agents/tools.py`: add a tool builder and include it in `build_framework_tools()`. The agent gets RAG + MCP tools by default.

## Adding LLM / RAG / Vector Store

- **LLM:** Implement `framework.llm.base.LLMClient` and wire in `framework.api.deps.get_llm`.
- **RAG / Vector store:** Implement `framework.rag.base.RAGClient` (e.g. ChromaRAG, PineconeRAG). Set `VECTOR_STORE` and backend env vars; `get_rag` in deps selects the backend.

This keeps the same structure as a **framework** with **separate components** and **clients** for onboarding, admin, tasks, agents, batch, queue, prompts, graph, and evaluation.
