"""Sidebar components for configuration and controls."""

import streamlit as st

from config import settings
from core.token_tracker import get_token_tracker
from ui.token_display import render_token_display_full
from ui.chat_controls import render_tool_selector


def render_sidebar():
    """Render the sidebar with configuration and controls."""
    with st.sidebar:
        st.title("⚙️ Settings")
        st.markdown("---")
        
        # Token usage display
        render_token_usage()
        
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


def render_token_usage():
    """Render token usage display in sidebar."""
    st.subheader("📈 Token Usage")
    
    if "workflow" in st.session_state:
        workflow = st.session_state.workflow
        session = workflow.session
        messages = session.get_messages()
        
        token_tracker = get_token_tracker()
        
        system_prompt = getattr(settings, 'SYSTEM_PROMPT', 'You are a helpful assistant.')
        context_docs = []
        
        usage = token_tracker.get_current_usage(
            messages=messages,
            system_prompt=system_prompt,
            context_docs=context_docs
        )
        
        render_token_display_full(usage, show_breakdown=True, show_warning=True)


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
        
    # Tool mode selector
    st.markdown("---")
    render_tool_mode_selector()


def render_tool_mode_selector():
    """Render tool mode selector in sidebar."""
    st.subheader("🛠️ Tool Mode")
    
    tool_modes = getattr(settings, 'TOOL_MODES', {
        "general": {"name": "General Chat", "system_prompt": "You are a helpful assistant."},
        "code": {"name": "Code Assistant", "system_prompt": "You are an expert programmer. Focus on code quality."},
        "document": {"name": "Document Analyzer", "system_prompt": "You analyze documents and extract key information."}
    })
    
    current_tool = st.session_state.get("current_tool", "general")
    
    selected_tool = render_tool_selector(
        current_tool=current_tool,
        tool_modes=tool_modes,
        on_change=lambda tool: _handle_tool_change(tool)
    )
    
    if selected_tool != current_tool:
        st.session_state.current_tool = selected_tool


def _handle_tool_change(tool: str):
    """Handle tool mode change."""
    st.session_state.current_tool = tool
    st.rerun()


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
