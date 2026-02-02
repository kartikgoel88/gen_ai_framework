"""Gen AI Learning: End-to-end flows, tutorials, and interactive exploration."""

import json
from pathlib import Path
from typing import Optional

import streamlit as st

from ui.services import (
    get_llm_client,
    get_doc_processor,
    get_rag_client,
    get_ocr_processor,
    save_uploaded_file,
)
from ui.components import (
    display_sources,
    display_chat_history,
    retrieve_rag_sources,
    parse_json_input,
    handle_extraction_result,
    safe_get_rag_client,
)


st.header("🎓 Gen AI Learning")
st.markdown("""
Explore complete end-to-end workflows and learn how to combine Gen AI components effectively.
Interactive tutorials, examples, and hands-on experimentation.
""")

# Initialize chat history
if "learning_chat_history" not in st.session_state:
    st.session_state.learning_chat_history = []

# Main tabs
tab_flows, tab_chatbot, tab_agents, tab_chains, tab_prompts, tab_graph = st.tabs([
    "📖 End-to-End Flows", "🤖 Document Q&A", "🔗 Agents", "⚡ Chains", "📝 Prompts", "📊 LangGraph"
])

# === END-TO-END FLOWS TAB ===
with tab_flows:
    st.subheader("Complete End-to-End Workflows")
    st.markdown("""
    Learn how to build complete AI applications by following these step-by-step workflows.
    Each flow demonstrates how multiple components work together.
    """)
    
    # Flow 1: Document Q&A System
    flow1 = st.expander("📚 Flow 1: Document Q&A System", expanded=True)
    with flow1:
        st.markdown("""
        **Goal**: Build a chatbot that answers questions from your documents
        
        **Workflow**:
        ```
        Documents/OCR → Extract Text → RAG Ingest → Query → LLM Answer
                              ↓
                        Embeddings (automatic)
        ```
        
        **Step-by-Step**:
        
        1. **Process Documents** (go to Document Q&A tab → Setup)
           - Upload PDFs, DOCX, or images
           - Extract text using document processor
           - For scanned docs, use OCR or Docling
        
        2. **Build Knowledge Base** (go to Document Q&A tab → Setup → RAG Ingest)
           - Copy extracted text
           - Add metadata (source, category, date)
           - Ingest into RAG store
           - Repeat for all documents
        
        3. **Query Your Knowledge** (go to Document Q&A tab → Chatbot)
           - Ask questions about your documents
           - Get answers with source citations
           - Chat history maintained
        
        **Try It**: Use the **Document Q&A** tab to follow this flow interactively!
        """)
    
    # Flow 2: Agent with Web Search
    flow2 = st.expander("🔍 Flow 2: Intelligent Agent with Web Search", expanded=False)
    with flow2:
        st.markdown("""
        **Goal**: Build an agent that can search documents AND the web
        
        **Workflow**:
        ```
        Question → Agent → [RAG Tool | Web Search Tool] → Answer
        ```
        
        **Step-by-Step**:
        
        1. **Setup RAG Knowledge Base** (same as Flow 1)
           - Process and ingest documents
        
        2. **Use Agent Mode** (go to Document Q&A tab → Enable Agent Mode)
           - Agent has access to:
             - **RAG Tool**: Search your documents
             - **Web Search Tool**: Search internet (LinkedIn, news, etc.)
             - **MCP Tools**: Additional tools if configured
        
        3. **Ask Complex Questions**
           - "Find LinkedIn profile for John Doe" → Uses web search
           - "What does our policy say about expenses?" → Uses RAG
           - "Based on our docs and current trends..." → Uses both
        
        **Try It**: Enable Agent Mode in the **Document Q&A** tab!
        """)
    
    # Flow 3: Batch Document Processing
    flow3 = st.expander("📦 Flow 3: Batch Document Processing", expanded=False)
    with flow3:
        st.markdown("""
        **Goal**: Process multiple documents efficiently
        
        **Workflow**:
        ```
        Documents → Extract → Batch LLM → Structured Output
        ```
        
        **Step-by-Step**:
        
        1. **Prepare Documents**
           - Collect multiple documents (PDFs, images, etc.)
           - Extract text from each
        
        2. **Batch Processing** (go to Chains tab → Batch)
           - Process multiple prompts at once
           - Use chains for structured extraction
           - Classify or categorize results
        
        3. **Post-Process Results**
           - Extract structured data
           - Ingest into RAG if needed
           - Generate reports
        
        **Use Cases**:
        - Process expense bills (see Batch Expense app)
        - Classify customer feedback
        - Extract data from forms
        """)
    
    # Flow 4: Chain-Based Processing
    flow4 = st.expander("⚡ Flow 4: Chain-Based Processing", expanded=False)
    with flow4:
        st.markdown("""
        **Goal**: Use pre-built chains for common tasks
        
        **Workflow**:
        ```
        Input → Chain → Structured Output
        ```
        
        **Available Chains**:
        
        - **RAG Chain**: Query knowledge base with context
        - **Summarization Chain**: Summarize long texts
        - **Classification Chain**: Categorize text
        - **Extraction Chain**: Extract structured data
        - **Structured Chain**: Get JSON outputs
        
        **Step-by-Step**:
        
        1. **Choose Chain Type** (go to Chains tab)
           - Select appropriate chain for your task
        
        2. **Configure Parameters**
           - Set templates, labels, schemas
        
        3. **Invoke Chain**
           - Provide input data
           - Get processed output
        
        **Try It**: Use the **Chains** tab to experiment!
        """)
    
    # Flow 5: Complete Application
    flow5 = st.expander("💼 Flow 5: Complete Application (Batch Expense)", expanded=False)
    with flow5:
        st.markdown("""
        **Goal**: Build a complete business application
        
        **Workflow**:
        ```
        Policy + Bills → Extract → LLM Analysis → Decisions
        ```
        
        **Components Used**:
        - Document processing
        - LLM for analysis
        - Structured extraction
        - Batch processing
        
        **See**: **Batch Expense** tab for complete implementation!
        """)

# === DOCUMENT Q&A TAB (from Tasks) ===
with tab_chatbot:
    st.subheader("Document Q&A Chatbot")
    st.markdown("""
    **Complete Flow**: Documents → Extract → RAG Ingest → Query
    
    Build a knowledge base from your documents and ask questions!
    """)
    
    # Setup and Chatbot tabs
    chatbot_tabs = st.tabs(["🤖 Chatbot", "⚙️ Setup"])
    
    # Chatbot
    with chatbot_tabs[0]:
        # Check if RAG is configured
        rag, rag_configured, rag_error = safe_get_rag_client()
        if not rag_configured:
            st.warning("⚠️ RAG is not configured. Please configure RAG settings to use the chatbot.")
        
        if rag_configured:
            # Mode selection
            use_agent = st.checkbox(
                "🤖 Use Agent Mode (RAG + Web Search)",
                value=False,
                help="Agent mode can search your documents AND the web (e.g., LinkedIn profiles). RAG mode only searches your documents."
            )
            
            # Display chat history
            display_chat_history(st.session_state.learning_chat_history)
            
            # Chat input
            placeholder_text = "Ask a question about your documents..."
            if use_agent:
                placeholder_text = "Ask a question (can search documents + web, e.g., 'Find LinkedIn profile for John Doe')..."
            
            user_question = st.chat_input(placeholder_text)
            
            if user_question:
                st.session_state.learning_chat_history.append({
                    "role": "user",
                    "content": user_question
                })
                
                with st.chat_message("user"):
                    st.write(user_question)
                
                try:
                    with st.chat_message("assistant"):
                        if use_agent:
                            with st.spinner("🤖 Agent is thinking and searching..."):
                                from ui.services import get_agent_client
                                agent = get_agent_client()
                                answer = agent.invoke(user_question)
                                
                                sources = retrieve_rag_sources(rag, user_question, top_k=4)
                                
                                st.write(answer)
                                st.caption("💡 Agent can search both your documents (RAG) and the web. Try asking: 'Find LinkedIn profile for [name]'")
                                
                                if sources:
                                    display_sources(sources, title="📄 Document Sources")
                        else:
                            with st.spinner("Thinking..."):
                                llm = get_llm_client()
                                answer = rag.query(user_question, llm_client=llm, top_k=4)
                                sources = retrieve_rag_sources(rag, user_question, top_k=4)
                                
                                st.write(answer)
                                display_sources(sources)
                        
                        st.session_state.learning_chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.exception(e)
            
            if st.button("🗑️ Clear Chat History"):
                st.session_state.learning_chat_history = []
                st.rerun()
        else:
            st.info("Go to **Setup** tab to process documents and build your knowledge base.")
    
    # Setup
    with chatbot_tabs[1]:
        st.write("**Process documents and build your knowledge base**")
        
        setup_subtabs = st.tabs(["📄 Documents", "🖼️ OCR", "📚 RAG Ingest", "🔗 Confluence"])
        
        with setup_subtabs[0]:
            doc_file = st.file_uploader("Document file", type=["pdf", "docx", "txt"], key="learning_doc_file")
            if doc_file:
                if st.button("Extract Text", key="learning_extract_btn"):
                    try:
                        doc = get_doc_processor()
                        path = save_uploaded_file(doc_file, "learning_docs")
                        if path:
                            with st.spinner("Extracting text..."):
                                result = doc.extract(path)
                            handle_extraction_result(result, "learning_extracted_text")
                    except Exception as e:
                        st.exception(e)
        
        with setup_subtabs[1]:
            ocr_file = st.file_uploader("Image file (PNG, JPG)", type=["png", "jpg", "jpeg"], key="learning_ocr_file")
            if ocr_file:
                if st.button("Run OCR", key="learning_ocr_btn"):
                    try:
                        ocr = get_ocr_processor()
                        content = ocr_file.read()
                        with st.spinner("Running OCR..."):
                            result = ocr.extract_from_bytes(content)
                        handle_extraction_result(result, "learning_ocr_text")
                    except Exception as e:
                        st.exception(e)
        
        with setup_subtabs[2]:
            rag_text = st.text_area("Text to ingest", height=200, placeholder="Paste extracted text here...", key="learning_rag_text")
            rag_metadata = st.text_input("Metadata (JSON, optional)", placeholder='{"source": "doc.pdf"}', key="learning_rag_metadata")
            
            if st.button("✅ Ingest into RAG", key="learning_rag_ingest_btn"):
                if rag_text:
                    try:
                        rag = get_rag_client()
                        metadata_dict, json_error = parse_json_input(rag_metadata)
                        if json_error:
                            st.warning("Invalid JSON metadata, ignoring.")
                        rag.add_documents([rag_text], metadatas=[metadata_dict] if metadata_dict else None)
                        st.success("✅ Text ingested successfully! You can now query it in the Chatbot tab.")
                    except Exception as e:
                        st.exception(e)
                else:
                    st.warning("Enter text to ingest.")
        
        with setup_subtabs[3]:
            st.write("**Ingest Confluence pages directly**")
            confluence_space = st.text_input("Space key (optional)", key="learning_confluence_space")
            confluence_pages = st.text_input("Page IDs (comma-separated)", key="learning_confluence_pages")
            if st.button("Ingest from Confluence", key="learning_confluence_btn"):
                if not confluence_space and not confluence_pages:
                    st.warning("Provide space_key and/or page_ids.")
                else:
                    try:
                        from src.framework.config import get_settings
                        from src.framework.api.deps import get_confluence_client
                        settings = get_settings()
                        confluence = get_confluence_client(settings)
                        if confluence is None:
                            st.error("Confluence not configured.")
                        else:
                            page_ids_list = [p.strip() for p in confluence_pages.split(",")] if confluence_pages else None
                            with st.spinner("Fetching and ingesting..."):
                                pages = confluence.fetch_pages_for_ingest(
                                    space_key=confluence_space or None,
                                    page_ids=page_ids_list,
                                    limit=100,
                                )
                            if pages:
                                rag = get_rag_client()
                                texts = [p[0] for p in pages]
                                metadatas = [p[1] for p in pages]
                                rag.add_documents(texts, metadatas=metadatas)
                                st.success(f"✅ Ingested {len(texts)} pages!")
                    except Exception as e:
                        st.exception(e)

# === AGENTS TAB ===
with tab_agents:
    st.subheader("Agent Playground")
    st.markdown("""
    **ReAct Agents** with RAG + Web Search + MCP tools.
    Agents can reason about which tools to use for each question.
    """)
    
    msg = st.text_area("Message", placeholder="Ask the agent something…", height=100)
    system_prompt = st.text_input("System prompt (optional)", key="learning_agent_sys")
    
    if st.button("Invoke Agent", key="learning_agent_btn"):
        if not (msg or "").strip():
            st.warning("Enter a message.")
        else:
            try:
                from ui.services import get_agent_client
                agent = get_agent_client()
                with st.spinner("🤖 Agent is thinking..."):
                    response = agent.invoke(msg.strip(), system_prompt=system_prompt or None)
                st.success("Agent Response:")
                st.write(response)
                st.info("💡 Try asking: 'Find LinkedIn profile for [name]' or 'What's the latest news about AI?'")
            except Exception as e:
                st.exception(e)
    
    st.divider()
    st.markdown("""
    **Agent Capabilities**:
    - **RAG Tool**: Searches your ingested documents
    - **Web Search Tool**: Searches the internet (LinkedIn, news, etc.)
    - **MCP Tools**: Additional tools if configured
    
    **Example Questions**:
    - "Find LinkedIn profile for John Doe"
    - "What does our policy say about expenses?" (if you've ingested policy docs)
    - "What's the latest news about AI?"
    """)

# === CHAINS TAB ===
with tab_chains:
    st.subheader("Chain Experimentation")
    st.markdown("""
    Experiment with different chain types for various tasks.
    """)
    
    chain_type = st.selectbox(
        "Chain type",
        [
            "rag", "prompt", "structured", "summarization", 
            "classification", "extraction",
            "langchain_prompt", "langchain_chat", "langchain_rag"
        ],
        key="learning_chain_type"
    )
    
    chain_inputs = st.text_area("Inputs (JSON)", placeholder='{"prompt": "Hello", "question": "What is AI?"}', height=100, key="learning_chain_inputs")
    
    if chain_type in ["prompt", "structured", "langchain_prompt", "summarization"]:
        chain_template = st.text_input("Template", placeholder="{prompt}", key="learning_chain_template")
    else:
        chain_template = None
    
    if chain_type in ["langchain_rag", "rag"]:
        chain_top_k = st.number_input("Top K", min_value=1, max_value=20, value=4, key="learning_chain_top_k")
    else:
        chain_top_k = None
    
    if chain_type == "classification":
        chain_labels = st.text_input("Labels (comma-separated)", placeholder="positive, negative, neutral", key="learning_chain_labels")
    else:
        chain_labels = None
    
    if chain_type == "extraction":
        chain_schema = st.text_input("Output schema", placeholder="title, author, date", key="learning_chain_schema")
    else:
        chain_schema = None
    
    if st.button("Invoke Chain", key="learning_chain_btn"):
        if chain_inputs:
            inputs_dict, json_error = parse_json_input(chain_inputs)
            if json_error:
                st.error(f"Invalid JSON for inputs: {json_error}")
            
            if inputs_dict:
                try:
                    llm = get_llm_client()
                    rag = get_rag_client()
                    
                    from src.framework.config import get_settings
                    from src.framework.chains import (
                        PromptChain, StructuredChain, SummarizationChain,
                        ClassificationChain, ExtractionChain,
                        build_langchain_prompt_chain, build_langchain_chat_prompt_chain, build_langchain_rag_chain,
                    )
                    from src.framework.api.deps import get_rag_chain
                    from src.framework.chains.summarization_chain import DEFAULT_SUMMARY_PROMPT
                    
                    with st.spinner("Invoking chain..."):
                        if chain_type == "rag":
                            rag_chain = get_rag_chain(get_settings())
                            output = rag_chain.invoke(inputs_dict)
                        elif chain_type == "prompt":
                            chain = PromptChain(llm=llm, template=chain_template or "{prompt}")
                            output = chain.invoke(inputs_dict)
                        elif chain_type == "structured":
                            chain = StructuredChain(llm=llm, template=chain_template or "{prompt}")
                            output = chain.invoke(inputs_dict)
                        elif chain_type == "summarization":
                            chain = SummarizationChain(llm=llm, prompt_template=chain_template or DEFAULT_SUMMARY_PROMPT)
                            output = chain.invoke(inputs_dict)
                        elif chain_type == "classification":
                            labels = [l.strip() for l in chain_labels.split(",")] if chain_labels else ["positive", "negative", "neutral"]
                            chain = ClassificationChain(llm=llm, labels=labels)
                            output = chain.invoke(inputs_dict)
                        elif chain_type == "extraction":
                            schema = chain_schema or "key facts and entities"
                            chain = ExtractionChain(llm=llm, schema=schema)
                            output = chain.invoke(inputs_dict)
                        else:
                            output = {"error": f"Unknown chain_type: {chain_type}"}
                    
                    st.success("Output:")
                    st.write(output)
                except Exception as e:
                    st.exception(e)
            else:
                st.warning("Enter inputs as JSON.")

# === PROMPTS TAB ===
with tab_prompts:
    st.subheader("Versioned Prompts")
    st.markdown("""
    View and manage versioned prompts from the framework store.
    """)
    
    store = None
    try:
        from src.framework.config import get_settings
        from src.framework.prompts.store import PromptStore
        settings = get_settings()
        base = getattr(settings, "PROMPTS_BASE_PATH", "./data/prompts")
        store = PromptStore(base_path=base)
        names = store.list_names()
    except Exception as e:
        names = []
        st.warning(f"Could not load prompt store: {e}")
    
    if store and names:
        name = st.selectbox("Prompt name", names)
        if name:
            versions = store.list_versions(name)
            version = st.selectbox("Version", versions or ["v1"]) if versions else "v1"
            p = store.get(name, version) if versions else store.get(name, "v1")
            if p:
                st.text_area("Body", value=p.body, height=200, disabled=True)
            else:
                st.info("No prompt found.")
    else:
        st.info("No prompts in store. Set PROMPTS_BASE_PATH and add prompt files.")

# === LANGGRAPH TAB ===
with tab_graph:
    st.subheader("LangGraph Workflows")
    st.markdown("""
    Experiment with LangGraph-based workflows (RAG graph or agent graph).
    """)
    
    query = st.text_input("Query", placeholder="Ask a question…")
    graph_type = st.radio("Graph type", ["rag", "agent"], horizontal=True)
    top_k = st.number_input("RAG top_k", min_value=1, max_value=20, value=4)
    
    if st.button("Invoke Graph", key="learning_graph_btn"):
        if not (query or "").strip():
            st.warning("Enter a query.")
        else:
            try:
                from src.framework.config import get_settings
                from src.framework.api.deps import get_llm, get_rag, get_mcp_client
                from src.framework.graph.rag_graph import build_rag_graph
                from src.framework.graph.agent_graph import build_agent_graph
                from langchain_openai import ChatOpenAI
                from langchain_core.messages import HumanMessage
                
                settings = get_settings()
                llm = get_llm(settings)
                rag = get_rag(settings)
                
                if graph_type == "rag":
                    graph = build_rag_graph(llm=llm, rag=rag, top_k=top_k)
                    result = graph.invoke({"query": query.strip()})
                    st.write(result.get("response", ""))
                    if result.get("context"):
                        with st.expander("Context"):
                            st.text(result["context"])
                else:
                    mcp = get_mcp_client(settings)
                    chat_llm = ChatOpenAI(
                        model=settings.LLM_MODEL,
                        temperature=settings.TEMPERATURE,
                        openai_api_key=settings.OPENAI_API_KEY,
                    )
                    graph = build_agent_graph(llm=chat_llm, rag=rag, mcp_client=mcp)
                    result = graph.invoke({"messages": [HumanMessage(content=query.strip())]})
                    messages = result.get("messages", [])
                    last = messages[-1] if messages else None
                    text = last.content if last and hasattr(last, "content") else str(result)
                    st.write(text)
            except Exception as e:
                st.exception(e)
