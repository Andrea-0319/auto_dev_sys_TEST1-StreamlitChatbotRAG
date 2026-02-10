"""Sidebar components for configuration and controls."""

import streamlit as st

from config import settings


def render_sidebar():
    """Render the sidebar with configuration and controls."""
    with st.sidebar:
        st.title("⚙️ Settings")
        st.markdown("---")
        
        # Model settings
        render_model_settings()
        
        st.markdown("---")
        
        # RAG settings
        render_rag_settings()
        
        st.markdown("---")
        
        # Session info
        render_session_info()
        
        st.markdown("---")
        
        # About
        render_about()


def render_model_settings():
    """Render model configuration section."""
    st.subheader("🤖 Model Settings")
    
    # Temperature
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=settings.TEMPERATURE,
        step=0.1,
        help="Higher values make output more random, lower more deterministic"
    )
    
    # Max tokens
    max_tokens = st.slider(
        "Max Tokens",
        min_value=128,
        max_value=2048,
        value=settings.MAX_TOKENS,
        step=128,
        help="Maximum number of tokens to generate"
    )
    
    # Store in session state for use
    st.session_state.temperature = temperature
    st.session_state.max_tokens = max_tokens


def render_rag_settings():
    """Render RAG configuration section."""
    st.subheader("🔍 RAG Settings")
    
    # Top-k retrieval
    top_k = st.slider(
        "Top-K Retrieval",
        min_value=1,
        max_value=10,
        value=settings.TOP_K_RETRIEVAL,
        step=1,
        help="Number of chunks to retrieve"
    )
    
    # Chunk size
    chunk_size = st.slider(
        "Chunk Size",
        min_value=256,
        max_value=1024,
        value=settings.CHUNK_SIZE,
        step=64,
        help="Size of text chunks"
    )
    
    st.session_state.top_k = top_k
    st.session_state.chunk_size = chunk_size


def render_session_info():
    """Render session information."""
    st.subheader("📊 Session Info")
    
    if "workflow" in st.session_state:
        workflow = st.session_state.workflow
        stats = workflow.get_stats()
        
        rag_stats = stats.get("rag", {})
        st.metric("Documents", rag_stats.get("document_count", 0))
        st.metric("Vectors", rag_stats.get("vector_count", 0))
        
        session_stats = stats.get("session", {})
        st.metric("Messages", session_stats.get("message_count", 0))


def render_about():
    """Render about section."""
    st.subheader("ℹ️ About")
    st.markdown(f"""
    **Model**: `{settings.MODEL_NAME}`
    
    **Embedding**: `{settings.EMBEDDING_MODEL}`
    
    **Device**: `{settings.DEVICE}`
    
    Built with ❤️ using Streamlit and Hugging Face
    """)
