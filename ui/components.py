"""Reusable UI components for Streamlit pages."""

import json
from typing import Any, Dict, List, Optional

import streamlit as st


def display_sources(sources: List[str], title: str = "📄 Sources", max_length: int = 500) -> None:
    """Display a list of sources in an expander.
    
    Args:
        sources: List of source text strings
        title: Title for the expander
        max_length: Maximum length to display per source (truncates with "...")
    """
    if not sources:
        return
    
    with st.expander(title):
        for j, source in enumerate(sources, 1):
            st.write(f"**Source {j}:**")
            display_text = source[:max_length] + "..." if len(source) > max_length else source
            st.code(display_text)


def display_chat_history(
    chat_history: List[Dict[str, Any]], 
    sources_key: str = "sources"
) -> None:
    """Display chat history with optional sources.
    
    Args:
        chat_history: List of message dicts with 'role' and 'content' keys
        sources_key: Key in message dict for sources list
    """
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get(sources_key):
                display_sources(message[sources_key])


def retrieve_rag_sources(rag_client, query: str, top_k: int = 4) -> List[str]:
    """Retrieve RAG sources for a query.
    
    Args:
        rag_client: RAG client instance
        query: Query string
        top_k: Number of chunks to retrieve
        
    Returns:
        List of source text strings
    """
    try:
        chunks = rag_client.retrieve(query, top_k=top_k)
        return [
            chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
            for chunk in chunks
            if chunk
        ]
    except Exception:
        return []


def parse_json_input(json_str: str, default: Optional[Any] = None) -> tuple[Optional[Any], Optional[str]]:
    """Parse JSON input string with error handling.
    
    Args:
        json_str: JSON string to parse
        default: Default value to return if parsing fails
        
    Returns:
        Tuple of (parsed_value, error_message). If successful, error_message is None.
    """
    if not json_str or not json_str.strip():
        return default, None
    
    try:
        parsed = json.loads(json_str)
        return parsed, None
    except json.JSONDecodeError as e:
        return default, f"Invalid JSON: {e}"


def display_json_parse_warning(error_msg: str) -> None:
    """Display a warning for JSON parsing errors."""
    if error_msg:
        st.warning(error_msg)


def handle_extraction_result(result, text_area_key: str, success_msg: str = "✅ Text extracted!") -> bool:
    """Handle document extraction result and display in UI.
    
    Args:
        result: Extraction result object with .text and optionally .error attributes
        text_area_key: Key for the text_area widget
        success_msg: Success message to display
        
    Returns:
        True if extraction was successful, False otherwise
    """
    if result.text:
        st.success(success_msg)
        st.text_area("Extracted text", value=result.text, height=200, key=text_area_key)
        st.info("💡 Copy the text above and paste it in the **RAG Ingest** tab.")
        return True
    else:
        reason = getattr(result, "error", "Unknown reason.")
        st.warning(f"No text could be extracted. **Reason:** {reason}")
        return False


def safe_get_rag_client():
    """Safely get RAG client, returning (client, is_configured, error_message).
    
    Returns:
        Tuple of (rag_client or None, bool, error_message or None)
    """
    try:
        from ui.services import get_rag_client
        rag = get_rag_client()
        return rag, True, None
    except Exception as e:
        return None, False, str(e)
