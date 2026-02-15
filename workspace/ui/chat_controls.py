"""Enhanced chat control UI components."""

import streamlit as st

from core.session import Message


def render_enhanced_controls(
    messages: list,
    on_clear: callable = None,
    on_summarize: callable = None,
    on_search: callable = None,
    on_export: callable = None
) -> None:
    """Render enhanced chat control panel.
    
    Args:
        messages: Current messages.
        on_clear: Callback for clear action.
        on_summarize: Callback for summarize action.
        on_search: Callback for search action.
        on_export: Callback for export action.
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🗑️ Clear", key="clear_chat_enhanced"):
            if on_clear:
                on_clear()
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("📝 Summarize", key="summarize_context"):
            if on_summarize:
                on_summarize()
    
    with col3:
        with st.popover("🔍 Search", use_container_width=True):
            search_query = st.text_input("Search messages:", key="search_input")
            if st.button("Search", key="do_search") and search_query:
                if on_search:
                    results = on_search(search_query)
                    if results:
                        st.success(f"Found {len(results)} messages")
                    else:
                        st.info("No messages found")
    
    with col4:
        with st.popover("💾 Export", use_container_width=True):
            export_format = st.selectbox(
                "Export Format",
                ["Markdown", "JSON", "Plain Text"]
            )
            
            if st.button("Download Export"):
                if on_export:
                    on_export(export_format.lower().replace(" ", "_"))


def render_message_actions(
    message: Message,
    index: int,
    on_copy: callable = None,
    on_delete: callable = None,
    on_pin: callable = None,
    on_feedback: callable = None,
    on_branch: callable = None
) -> None:
    """Render action buttons for a single message.
    
    Args:
        message: Message object.
        index: Message index.
        on_copy: Callback for copy action.
        on_delete: Callback for delete action.
        on_pin: Callback for pin action.
        on_feedback: Callback for feedback action.
        on_branch: Callback for branch action.
    """
    with st.expander("⚙️ Actions", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("📋 Copy", key=f"copy_{index}"):
                if on_copy:
                    on_copy(message.content)
                st.success("Copied!")
        
        with col2:
            pin_text = "📌 Unpin" if message.is_pinned else "📌 Pin"
            if st.button(pin_text, key=f"pin_{index}"):
                if on_pin:
                    on_pin(message.id, not message.is_pinned)
                st.rerun()
        
        with col3:
            if st.button("🗑️ Delete", key=f"delete_{index}"):
                if on_delete:
                    on_delete(message.id)
                st.rerun()
        
        with col4:
            if message.role == "assistant":
                fb_col1, fb_col2 = st.columns(2)
                
                with fb_col1:
                    if st.button("👍", key=f"thumb_up_{index}"):
                        if on_feedback:
                            on_feedback(message.id, "positive")
                        st.rerun()
                
                with fb_col2:
                    if st.button("👎", key=f"thumb_down_{index}"):
                        if on_feedback:
                            on_feedback(message.id, "negative")
                        st.rerun()
        
        with col5:
            if message.role == "user" and on_branch:
                if st.button("🔀 Branch", key=f"branch_msg_{index}"):
                    on_branch(index)


def render_search_results(results: list, on_select: callable = None) -> None:
    """Render search results.
    
    Args:
        results: List of matching messages.
        on_select: Callback when result is selected.
    """
    if not results:
        st.info("No results found")
        return
    
    st.markdown(f"**Found {len(results)} results:**")
    
    for i, msg in enumerate(results):
        with st.container():
            col1, col2 = st.columns([1, 4])
            
            with col1:
                icon = "🧑" if msg.role == "user" else "🤖"
                st.markdown(f"{icon} *{msg.role.title()}*")
            
            with col2:
                preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                st.markdown(f"_{msg.timestamp.strftime('%H:%M')}_: {preview}")
            
            if on_select:
                if st.button(f"Go to message {i+1}", key=f"goto_{msg.id}"):
                    on_select(msg)
            
            st.markdown("---")


def render_tool_selector(
    current_tool: str,
    tool_modes: dict,
    on_change: callable = None
) -> str:
    """Render tool mode selector.
    
    Args:
        current_tool: Current tool mode.
        tool_modes: Dictionary of tool modes.
        on_change: Callback when tool changes.
        
    Returns:
        Selected tool ID.
    """
    tool_options = list(tool_modes.keys())
    tool_names = [tool_modes[t]["name"] for t in tool_options]
    
    current_index = tool_options.index(current_tool) if current_tool in tool_options else 0
    
    selected = st.selectbox(
        "🛠️ Tool Mode",
        options=range(len(tool_options)),
        format_func=lambda x: tool_names[x],
        index=current_index,
        key="tool_selector"
    )
    
    selected_tool = tool_options[selected]
    
    if selected_tool != current_tool and on_change:
        on_change(selected_tool)
    
    return selected_tool


def render_keyboard_shortcuts() -> None:
    """Render keyboard shortcuts help."""
    with st.expander("Keyboard Shortcuts (Coming Soon)"):
        st.markdown("""
        *Keyboard shortcuts are planned for future versions.*
        
        - **Ctrl+Enter**: Send message *(planned)*
        - **Ctrl+L**: Clear conversation *(planned)*
        - **Ctrl+S**: Summarize context *(planned)*
        - **Ctrl+E**: Export conversation *(planned)*
        """)


def render_confirmation_dialog(
    title: str,
    message: str,
    confirm_label: str = "Confirm",
    cancel_label: str = "Cancel"
) -> bool:
    """Render a confirmation dialog.
    
    Args:
        title: Dialog title.
        message: Confirmation message.
        confirm_label: Confirm button label.
        cancel_label: Cancel button label.
        
    Returns:
        True if confirmed.
    """
    with st.container():
        st.warning(f"**{title}**")
        st.write(message)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(confirm_label, key="confirm_action"):
                return True
        
        with col2:
            if st.button(cancel_label, key="cancel_action"):
                return False
        
        return False
