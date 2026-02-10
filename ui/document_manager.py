"""Document management UI components."""

from typing import Callable, List

import streamlit as st

from config import settings
from ui.components import render_info_box, render_empty_state


def render_document_upload(on_upload: Callable = None):
    """Render document upload section.
    
    Args:
        on_upload: Callback when file is uploaded.
    """
    st.subheader("📄 Upload Documents")
    
    uploaded_file = st.file_uploader(
        "Upload PDF, TXT, or Markdown files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=False,
        key="document_uploader"
    )
    
    if uploaded_file is not None:
        # Check file size
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        
        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            st.error(f"File too large ({file_size_mb:.1f}MB). Max size: {settings.MAX_FILE_SIZE_MB}MB")
            return
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.text(f"📄 {uploaded_file.name} ({file_size_mb:.2f} MB)")
        
        with col2:
            if st.button("📤 Upload", key="upload_button"):
                with st.spinner("Processing document..."):
                    if on_upload:
                        success, message = on_upload(uploaded_file)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)


def render_document_list(documents: List[dict], on_delete: Callable = None):
    """Render list of uploaded documents.
    
    Args:
        documents: List of document dictionaries.
        on_delete: Callback when document is deleted.
    """
    st.subheader("📚 Knowledge Base")
    
    if not documents:
        render_empty_state("No documents uploaded yet", icon="📭")
        return
    
    st.write(f"**{len(documents)} document(s) in knowledge base**")
    
    for doc in documents:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                file_type_icon = {
                    "pdf": "📕",
                    "txt": "📄",
                    "md": "📝"
                }.get(doc.get("type", "").lower(), "📄")
                
                st.text(f"{file_type_icon} {doc.get('name', 'Unknown')}")
                st.caption(f"{doc.get('chunk_count', 0)} chunks • {doc.get('size_mb', 0):.2f} MB")
            
            with col2:
                if st.button("🗑️", key=f"delete_{doc.get('id')}"):
                    if on_delete:
                        with st.spinner("Removing..."):
                            success = on_delete(doc.get('id'))
                            if success:
                                st.success("Removed!")
                                st.rerun()
                            else:
                                st.error("Failed to remove")
            
            with col3:
                # Status indicator
                st.markdown("✅ Active")
        
        st.divider()


def render_document_stats(stats: dict):
    """Render document statistics.
    
    Args:
        stats: Statistics dictionary.
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Documents", stats.get("document_count", 0))
    
    with col2:
        st.metric("Chunks", stats.get("vector_count", 0))
    
    with col3:
        st.metric("Dimension", stats.get("dimension", 384))


def render_clear_all_button(on_clear_all: Callable = None):
    """Render clear all documents button.
    
    Args:
        on_clear_all: Callback when clear all is clicked.
    """
    if st.button("🗑️ Clear All Documents", type="secondary", key="clear_all_docs"):
        confirm = st.checkbox("Confirm deletion?", key="confirm_clear")
        
        if confirm:
            if on_clear_all:
                with st.spinner("Clearing..."):
                    success = on_clear_all()
                    if success:
                        st.success("All documents cleared!")
                        st.rerun()
                    else:
                        st.error("Failed to clear documents")
