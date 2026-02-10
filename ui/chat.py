"""Chat interface components."""

import time
from typing import List, Optional

import streamlit as st

from config import settings
from core.session import Message
from ui.components import render_info_box


def render_chat_interface(messages: List[Message]):
    """Render the chat message history.
    
    Args:
        messages: List of messages to display.
    """
    for msg in messages:
        if msg.role == "user":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(msg.content)
        elif msg.role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                # Display main content
                st.markdown(msg.content)
                
                # Display sources if available
                if msg.sources:
                    source_text = ", ".join(msg.sources[:3])
                    if len(msg.sources) > 3:
                        source_text += f" (+{len(msg.sources) - 3} more)"
                    st.caption(f"📚 Sources: {source_text}")


def render_chat_input(disabled: bool = False) -> Optional[str]:
    """Render chat input field.
    
    Args:
        disabled: Whether input is disabled.
        
    Returns:
        User input or None.
    """
    return st.chat_input(
        "Ask a question...",
        disabled=disabled,
        key="chat_input"
    )


def render_thinking_indicator():
    """Render a thinking indicator."""
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        dots = [".", "..", "..."]
        for i in range(3):
            placeholder.markdown(f"💭 Thinking{dots[i % 3]}")
            time.sleep(0.3)
        placeholder.empty()


def render_streaming_response(generator, on_complete=None):
    """Render a streaming response.
    
    Args:
        generator: Generator yielding tokens.
        on_complete: Callback when complete.
        
    Returns:
        Full response text.
    """
    full_response = ""
    
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        
        try:
            for update in generator:
                update_type = update.get("type", "")
                
                if update_type == "token":
                    full_response = update.get("partial_response", "")
                    message_placeholder.markdown(full_response + "▌")
                    
                elif update_type == "status":
                    message_placeholder.markdown(f"⏳ {update.get('message', '')}")
                    
                elif update_type == "retrieval":
                    # Briefly show retrieval info
                    chunks = update.get("chunks", [])
                    if chunks:
                        sources = update.get("sources", [])
                        status_text = f"Found {len(chunks)} relevant chunks"
                        if sources:
                            status_text += f" from {len(sources)} documents"
                        message_placeholder.markdown(f"🔍 {status_text}")
                        time.sleep(0.5)
                    
                elif update_type == "complete":
                    final_response = update.get("response", full_response)
                    message_placeholder.markdown(final_response)
                    full_response = final_response
                    
                    # Show reasoning expander if available
                    reasoning = update.get("reasoning")
                    if reasoning:
                        with st.expander("💭 Show Reasoning Process", expanded=False):
                            st.markdown(reasoning)
                    
                    # Show sources
                    sources = update.get("sources", [])
                    if sources:
                        st.caption(f"📚 Sources: {', '.join(sources[:3])}")
                    
                    if on_complete:
                        on_complete(update)
                        
                elif update_type == "error":
                    error_msg = update.get("message", "An error occurred")
                    message_placeholder.error(error_msg)
                    st.session_state.error_message = error_msg
                    
        except Exception as e:
            message_placeholder.error(f"Error: {str(e)}")
            st.session_state.error_message = str(e)
    
    return full_response


def render_chat_controls(on_clear=None):
    """Render chat control buttons.
    
    Args:
        on_clear: Callback for clear button.
    """
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if st.button("🗑️ Clear", key="clear_chat"):
            if on_clear:
                on_clear()
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("💾 Export", key="export_chat"):
            export_chat_history()


def export_chat_history():
    """Export chat history to file."""
    if "messages" not in st.session_state or not st.session_state.messages:
        render_info_box("No messages to export", "warning")
        return
    
    messages = st.session_state.messages
    export_text = "# Chat History\n\n"
    
    for msg in messages:
        role_icon = "🧑" if msg.role == "user" else "🤖"
        export_text += f"{role_icon} **{msg.role.title()}**\n\n"
        export_text += f"{msg.content}\n\n"
        if hasattr(msg, 'reasoning') and msg.reasoning:
            export_text += f"*Reasoning: {msg.reasoning[:200]}...*\n\n"
        export_text += "---\n\n"
    
    st.download_button(
        label="📥 Download Chat",
        data=export_text,
        file_name="chat_history.md",
        mime="text/markdown",
        key="download_chat"
    )


def init_chat_state():
    """Initialize chat-related session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "streaming" not in st.session_state:
        st.session_state.streaming = False
    if "error_message" not in st.session_state:
        st.session_state.error_message = None
