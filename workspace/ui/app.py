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
from ui.chat_controls import render_enhanced_controls, render_tool_selector, render_confirmation_dialog
from core.summarization import get_summarization_service
from core.export import get_export_service


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


def handle_summarize():
    """Handle context summarization with confirmation dialog."""
    workflow = st.session_state.workflow
    messages = workflow.session.get_messages()
    
    if len(messages) < 4:
        st.info("Not enough messages to summarize (need at least 4)")
        return
    
    message_count = len([m for m in messages if m.role in ("user", "assistant")])
    
    with st.form("summarize_confirm_form"):
        st.warning(f"This will summarize {message_count} messages into a single summary message. The original messages will be replaced.")
        col1, col2 = st.columns(2)
        with col1:
            confirm = st.form_submit_button("Confirm Summarize")
        with col2:
            cancel = st.form_submit_button("Cancel")
        
        if confirm:
            _do_summarize(workflow)
        elif cancel:
            st.info("Summarization cancelled")

def _do_summarize(workflow):
    """Execute the actual summarization."""
    summarization_service = get_summarization_service(workflow.model_handler)
    
    if summarization_service:
        messages = workflow.session.get_messages()
        result = summarization_service.summarize_messages(messages)
        
        if result.new_message:
            for msg_id in result.original_message_ids:
                workflow.session.delete_message(msg_id)
            
            workflow.session._messages.append(result.new_message)
            
            st.session_state.last_summarize_result = {
                "count": len(result.original_message_ids),
                "tokens_saved": result.tokens_saved,
                "summary": result.summary[:200] + "..." if len(result.summary) > 200 else result.summary
            }
            st.success(f"Summarized {len(result.original_message_ids)} messages, saved ~{result.tokens_saved} tokens")
        else:
            st.info("Not enough messages to summarize")
    else:
        st.error("Summarization service not available")


def handle_export(export_format: str):
    """Handle conversation export."""
    workflow = st.session_state.workflow
    messages = workflow.session.get_messages()
    export_service = get_export_service()
    
    if not messages:
        st.info("No messages to export")
        return
    
    if export_format == "markdown":
        content = export_service.export_markdown(messages, include_metadata=True)
        file_name = "chat_history.md"
        mime = "text/markdown"
    elif export_format == "json":
        content = export_service.export_json(messages, include_metadata=True)
        file_name = "chat_history.json"
        mime = "application/json"
    elif export_format == "plain_text":
        content = export_service.export_plain_text(messages)
        file_name = "chat_history.txt"
        mime = "text/plain"
    else:
        st.error(f"Unknown export format: {export_format}")
        return
    
    st.download_button(
        label=f"Download {export_format.replace('_', ' ').title()}",
        data=content,
        file_name=file_name,
        mime=mime,
        key=f"download_{export_format}"
    )


def handle_search(query: str):
    """Handle message search."""
    workflow = st.session_state.workflow
    return workflow.session.search_messages(query)


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
        render_chat_interface(messages, st.session_state.workflow.session)
        
        # Chat input
        user_input = render_chat_input(disabled=st.session_state.get("streaming", False))
        
        if user_input:
            handle_query(user_input)
            st.rerun()
        
        # Chat controls - Enhanced
        st.markdown("---")
        render_enhanced_controls(
            messages=messages,
            on_clear=lambda: st.session_state.workflow.clear_conversation(),
            on_summarize=handle_summarize,
            on_search=handle_search,
            on_export=handle_export
        )
        
        # Show collapsible summary section if summarization was performed
        if "last_summarize_result" in st.session_state:
            result = st.session_state.last_summarize_result
            with st.expander(f"Last Summary ({result['count']} messages condensed, ~{result['tokens_saved']} tokens saved)", expanded=False):
                st.markdown(f"**Summary:**\n\n{result['summary']}")
    
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
