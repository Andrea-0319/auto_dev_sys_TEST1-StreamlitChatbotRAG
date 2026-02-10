"""Main Streamlit application."""

import sys
from pathlib import Path

# Add workspace to path
workspace_path = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_path))

import streamlit as st

from config import settings
from core.workflow import WorkflowEngine
from ui.components import render_header, render_footer, render_info_box
from ui.chat import (
    render_chat_interface,
    render_chat_input,
    render_streaming_response,
    render_chat_controls,
    init_chat_state,
)
from ui.document_manager import (
    render_document_upload,
    render_document_list,
    render_clear_all_button,
    render_document_stats,
)
from ui.sidebar import render_sidebar


def init_session_state():
    """Initialize Streamlit session state."""
    init_chat_state()
    
    if "workflow" not in st.session_state:
        st.session_state.workflow = WorkflowEngine()
    
    if "documents" not in st.session_state:
        st.session_state.documents = []


def handle_file_upload(uploaded_file):
    """Handle file upload.
    
    Args:
        uploaded_file: Streamlit uploaded file object.
        
    Returns:
        Tuple of (success, message).
    """
    workflow = st.session_state.workflow
    
    # Save file temporarily
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        success, message = workflow.upload_document(tmp_path, uploaded_file.getvalue())
        
        if success:
            # Refresh document list
            st.session_state.documents = workflow.get_document_list()
        
        return success, message
        
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def handle_document_delete(doc_id):
    """Handle document deletion.
    
    Args:
        doc_id: Document ID.
        
    Returns:
        True if successful.
    """
    workflow = st.session_state.workflow
    success = workflow.remove_document(doc_id)
    
    if success:
        st.session_state.documents = workflow.get_document_list()
    
    return success


def handle_clear_all():
    """Handle clear all documents."""
    workflow = st.session_state.workflow
    success = workflow.clear_all_documents()
    
    if success:
        st.session_state.documents = []
    
    return success


def handle_query(query: str):
    """Handle user query.
    
    Args:
        query: User query.
    """
    workflow = st.session_state.workflow
    
    # Process query
    generator = workflow.process_query(query, stream=True)
    
    # Render streaming response
    render_streaming_response(generator)
    
    # Refresh document list if needed
    st.session_state.documents = workflow.get_document_list()


def main():
    """Main application."""
    # Page config
    st.set_page_config(
        page_title=settings.PAGE_TITLE,
        page_icon=settings.PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    render_header()
    
    # Create tabs
    tab1, tab2 = st.tabs(["💬 Chat", "📄 Documents"])
    
    with tab1:
        # Chat interface
        st.markdown("### Chat with your documents")
        
        # Display messages
        messages = st.session_state.workflow.session.get_messages()
        render_chat_interface(messages)
        
        # Chat input
        user_input = render_chat_input(disabled=st.session_state.get("streaming", False))
        
        if user_input:
            handle_query(user_input)
            st.rerun()
        
        # Chat controls
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.workflow.clear_conversation()
                st.rerun()
        
        with col2:
            if st.button("💾 Export"):
                # Export handled in callback
                pass
    
    with tab2:
        # Document management
        col1, col2 = st.columns([1, 1])
        
        with col1:
            render_document_upload(on_upload=handle_file_upload)
        
        with col2:
            # Document stats
            workflow = st.session_state.workflow
            stats = workflow.get_stats()
            render_document_stats(stats.get("rag", {}))
        
        st.markdown("---")
        
        # Document list
        render_document_list(
            st.session_state.documents,
            on_delete=handle_document_delete
        )
        
        # Clear all button
        if st.session_state.documents:
            render_clear_all_button(on_clear_all=handle_clear_all)
    
    # Footer
    render_footer()


if __name__ == "__main__":
    main()
