# Framework Examples

This directory contains example scripts demonstrating how to use the framework.

## Examples

### Core Examples

#### basic_rag.py
Demonstrates basic RAG (Retrieval-Augmented Generation) usage:
- Creating a RAG client
- Ingesting documents
- Querying the knowledge base
- Using LLM for answer generation

**Usage:**
```bash
export OPENAI_API_KEY="your-key"
python examples/basic_rag.py
```

#### multi_provider_llm.py
Demonstrates using multiple LLM providers:
- Using the provider registry
- Switching between providers
- Adding custom providers

**Usage:**
```bash
export OPENAI_API_KEY="your-key"  # or XAI_API_KEY, GOOGLE_API_KEY, etc.
python examples/multi_provider_llm.py
```

#### chains_example.py
Demonstrates all chain types:
- Prompt Chain
- RAG Chain
- Structured Chain
- Summarization Chain
- Classification Chain
- Extraction Chain
- Pipeline (Multi-step chains)
- LangChain Integration

**Usage:**
```bash
export OPENAI_API_KEY="your-key"
python examples/chains_example.py
```

#### pipeline_example.py
Demonstrates multi-step chain composition:
- Simple pipelines (Extract → Summarize)
- Complex pipelines (Extract → Summarize → Classify)
- RAG pipelines (Retrieve → Extract → Structure)
- Conditional processing
- Custom output keys

**Usage:**
```bash
export OPENAI_API_KEY="your-key"
python examples/pipeline_example.py
```

#### adapters_example.py
Demonstrates LangChain adapter usage:
- Basic adapter usage
- Provider-specific adapters
- Message serialization
- Tool call parsing
- Using adapters in LangChain chains
- Streaming with adapters

**Usage:**
```bash
export OPENAI_API_KEY="your-key"
python examples/adapters_example.py
```

#### graph_example.py
Demonstrates LangGraph workflows:
- RAG Graph (Retrieve → Generate)
- Agent Graph (ReAct with tools)
- Streaming graph execution
- State inspection
- Multiple queries with state

**Usage:**
```bash
export OPENAI_API_KEY="your-key"
python examples/graph_example.py
```

#### agent_flow_example.py
Demonstrates the complete agent flow:
- Step-by-step flow visualization
- Simple questions (no tools)
- Questions requiring tools
- Multi-turn conversations
- Component explanation

**Usage:**
```bash
export OPENAI_API_KEY="your-key"
python examples/agent_flow_example.py
```

#### document_processing.py
Demonstrates document extraction:
- PDF, DOCX, TXT, Excel processing
- OCR for images
- File handling

**Usage:**
```bash
python examples/document_processing.py
```

### Agent Examples

#### agent_with_tools.py
Demonstrates agent usage with tools:
- Creating an agent with RAG and web search tools
- Automatic tool selection
- Answering questions using multiple tools

**Usage:**
```bash
export OPENAI_API_KEY="your-key"
python examples/agent_with_tools.py
```

#### streaming_agent.py
Demonstrates streaming agent reasoning:
- Real-time streaming of intermediate steps
- Tool selection and execution visibility
- Progressive response chunks

**Usage:**
```bash
export OPENAI_API_KEY="your-key"
python examples/streaming_agent.py
```

#### agent_with_memory.py
Demonstrates agents with persistent memory:
- Long-term conversation memory
- Semantic search over past conversations
- User-specific memory contexts

**Usage:**
```bash
export OPENAI_API_KEY="your-key"
python examples/agent_with_memory.py
```

#### multi_agent_example.py
Demonstrates multi-agent systems:
- Multiple specialized agents
- Agent orchestration
- Task delegation

**Usage:**
```bash
export OPENAI_API_KEY="your-key"
python examples/multi_agent_example.py
```

#### planning_agent_example.py
Demonstrates planning agents:
- Plan-and-solve pattern
- Step-by-step planning
- Automatic plan revision

**Usage:**
```bash
export OPENAI_API_KEY="your-key"
python examples/planning_agent_example.py
```

#### complete_agent_features.py
Demonstrates combining multiple agent features:
- Memory
- Monitoring
- Cost Tracking
- Personas
- Error Recovery

**Usage:**
```bash
export OPENAI_API_KEY="your-key"
python examples/complete_agent_features.py
```

#### persona_example.py
Demonstrates agent personas:
- Pre-defined personas (Researcher, Writer, Analyst, etc.)
- Custom persona creation
- Role-based behavior

**Usage:**
```bash
export OPENAI_API_KEY="your-key"
python examples/persona_example.py
```

#### error_recovery_example.py
Demonstrates error recovery:
- Automatic retry on failure
- Exponential backoff
- Error classification
- Fallback strategies

**Usage:**
```bash
export OPENAI_API_KEY="your-key"
python examples/error_recovery_example.py
```

### Evaluation & Monitoring

#### evaluation_example.py
Demonstrates agent evaluation:
- Task-based evaluation
- Multiple metrics
- Performance summaries

**Usage:**
```bash
export OPENAI_API_KEY="your-key"
python examples/evaluation_example.py
```

#### cost_monitoring_example.py
Demonstrates cost tracking and monitoring:
- Cost per operation
- Budget management
- Performance metrics
- Tool usage tracking

**Usage:**
```bash
export OPENAI_API_KEY="your-key"
python examples/cost_monitoring_example.py
```

## CLI Tools

The framework also provides CLI tools for common operations:

### RAG CLI (`rag-cli`)
```bash
# Ingest documents
rag-cli ingest --file document.txt
rag-cli ingest --text "Document text" --metadata '{"source": "doc1"}'

# Query RAG
rag-cli query "What is Python?" --top-k 5
rag-cli query "What is Python?" --llm  # Use LLM for answer

# Clear RAG store
rag-cli clear

# Export corpus
rag-cli export --output corpus.jsonl --format jsonl
```

### Agent CLI (`agent-cli`)
```bash
# Invoke agent
agent-cli invoke --message "Hello!"
agent-cli invoke --file question.txt --output response.txt

# List tools
agent-cli tools
agent-cli tools --json
```

### Chain CLI (`chain-cli`)
```bash
# Invoke chain
chain-cli invoke rag --inputs-json '{"question": "What is AI?"}'
chain-cli invoke prompt --inputs-file inputs.json --output result.json
```

### Batch CLI (`batch-run`)
```bash
# Process batch bills
batch-run --policy policy.pdf --folders ./bills/ -o results.json
batch-run --policy policy.pdf --zip bills.zip -o results.json
```

## Requirements

All examples require:
- Python 3.10+
- Framework installed: `pip install -e .`
- Appropriate API keys set in environment variables

## Additional Resources

- `README.md` - Main project documentation with API examples
- `docs/ARCHITECTURE.md` - Architecture documentation
- `docs/AGENTIC_AI_FEATURES.md` - Agent features guide
- `ui/pages/` - Streamlit UI examples
- `tests/` - Test files with usage patterns
