"""Home / Dashboard - Feature Overview."""

import streamlit as st

st.title("🤖 Gen AI Framework")
st.markdown("""
A **modular Gen AI framework** with separate components and domain clients. 
Build powerful AI applications with LLM, RAG, agents, chains, document processing, and more.
""")

st.divider()

# Core Framework Components
st.header("🔧 Core Framework Components")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🤖 LLM & Chat")
    st.markdown("""
    - **Multiple Providers**: OpenAI, Grok (xAI), Gemini, HuggingFace
    - **Streaming Support**: Real-time response streaming
    - **Batch Inference**: Process multiple prompts efficiently
    - **Chat Completion**: Simple Q&A interface
    """)

with col2:
    st.subheader("📚 RAG & Knowledge")
    st.markdown("""
    - **Vector Stores**: ChromaDB, Pinecone, Weaviate, Qdrant, pgvector
    - **Hybrid Search**: Vector + BM25 for better retrieval
    - **Reranking**: Cross-encoder reranking support
    - **Embeddings**: OpenAI, SentenceTransformer
    """)

with col3:
    st.subheader("🔗 Agents & Tools")
    st.markdown("""
    - **ReAct Agents**: Tool-using agents with reasoning
    - **RAG Tool**: Search knowledge base
    - **Web Search**: Search internet (including LinkedIn)
    - **MCP Tools**: Model Context Protocol integration
    """)

st.divider()

col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("📄 Document Processing")
    st.markdown("""
    - **Extract Text**: PDF, DOCX, TXT, Excel
    - **OCR**: EasyOCR for images and scanned docs
    - **Docling**: Layout-aware parsing + OCR
    - **LangChain Loaders**: PyPDF, Docx2txt, CSV
    """)

with col5:
    st.subheader("⚡ Chains & Pipelines")
    st.markdown("""
    - **RAG Chain**: Document Q&A
    - **Structured Chain**: JSON output extraction
    - **Summarization**: Text summarization
    - **Classification**: Text categorization
    - **Extraction**: Structured data extraction
    """)

with col6:
    st.subheader("📊 Advanced Features")
    st.markdown("""
    - **LangGraph**: Graph-based workflows
    - **Versioned Prompts**: A/B testing support
    - **Task Queue**: Celery for async processing
    - **Observability**: LLM tracing, evaluation
    - **Confluence**: Direct knowledge base ingestion
    """)

st.divider()

# Domain Clients / Applications
st.header("🚀 Domain Clients & Applications")

col7, col8 = st.columns(2)

with col7:
    st.subheader("💼 Batch Expense Processing")
    st.markdown("""
    **Purpose**: Process expense bills against company policy
    
    **Features**:
    - Upload policy documents
    - Process individual bills or ZIP folders
    - Automatic approve/reject decisions
    - Policy parsing and structured extraction
    - Client address validation
    
    **Use Cases**:
    - Expense reimbursement workflows
    - Policy compliance checking
    - Batch document processing
    """)

with col8:
    st.subheader("🎓 Gen AI Learning & Exploration")
    st.markdown("""
    **Purpose**: Learn and explore Gen AI capabilities
    
    **Features**:
    - Document Q&A chatbot (RAG + Web Search)
    - End-to-end workflow tutorials
    - Interactive agent playground
    - Chain experimentation
    - Complete flow demonstrations
    
    **Use Cases**:
    - Building knowledge bases
    - Learning RAG workflows
    - Exploring agent capabilities
    - Understanding AI patterns
    """)

st.divider()

# Key Workflows
st.header("🔄 Key Workflows")

workflow1, workflow2, workflow3 = st.columns(3)

with workflow1:
    st.markdown("""
    **📚 Document Q&A**
    ```
    Documents → Extract → RAG Ingest → Query
    ```
    - Upload and process documents
    - Build knowledge base
    - Ask questions with citations
    """)

with workflow2:
    st.markdown("""
    **🤖 Agent with Web Search**
    ```
    Question → Agent → RAG + Web Search → Answer
    ```
    - Search documents (RAG)
    - Search web (LinkedIn, news, etc.)
    - Intelligent tool selection
    """)

with workflow3:
    st.markdown("""
    **💼 Batch Processing**
    ```
    Policy + Bills → Extract → LLM → Decisions
    ```
    - Process multiple documents
    - Apply business rules
    - Generate structured outputs
    """)

st.divider()

# Quick Start
st.header("🚀 Quick Start")

st.markdown("""
1. **Explore Features**: Navigate to **Gen AI Learning** to see end-to-end workflows
2. **Try Applications**: Check out **Batch Expense** for a complete app example
3. **Build Your Own**: Use the framework components to build custom applications
""")

st.info("💡 **Tip**: Start with the **Gen AI Learning** tab to see complete end-to-end flows and learn how to combine components effectively.")

st.divider()

# Configuration
st.header("⚙️ Configuration")

try:
    from src.framework.config import get_settings
    s = get_settings()
    with st.expander("Framework Configuration (read-only)"):
        config_col1, config_col2 = st.columns(2)
        with config_col1:
            st.json({
                "LLM_PROVIDER": s.LLM_PROVIDER,
                "LLM_MODEL": s.LLM_MODEL,
                "VECTOR_STORE": s.VECTOR_STORE,
                "UPLOAD_DIR": s.UPLOAD_DIR,
            })
        with config_col2:
            st.json({
                "TEMPERATURE": s.TEMPERATURE,
                "CHUNK_SIZE": getattr(s, "CHUNK_SIZE", "default"),
                "CHUNK_OVERLAP": getattr(s, "CHUNK_OVERLAP", "default"),
            })
except Exception as e:
    st.warning(f"Config: {e}")
