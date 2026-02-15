"""Branch selector UI components for conversation branching."""

import streamlit as st

from core.session import Branch


def render_branch_dropdown(branches: list, current_branch_id: str = None) -> str:
    """Render branch dropdown selector.
    
    Args:
        branches: List of branches.
        current_branch_id: Currently active branch ID.
        
    Returns:
        Selected branch ID or None.
    """
    if not branches:
        return None
    
    branch_options = ["Main Conversation"] + [
        f"{b.name} ({b.message_count} messages)" 
        for b in branches
    ]
    branch_ids = [None] + [b.id for b in branches]
    
    current_index = 0
    if current_branch_id:
        try:
            current_index = branch_ids.index(current_branch_id) + 1
        except ValueError:
            current_index = 0
    
    selected = st.selectbox(
        "🌿 Conversation Branch",
        options=range(len(branch_options)),
        format_func=lambda x: branch_options[x],
        index=current_index,
        key="branch_selector"
    )
    
    return branch_ids[selected] if selected > 0 else None


def render_branch_tree(branches: list, current_branch_id: str = None) -> None:
    """Render branch tree visualization.
    
    Args:
        branches: List of branches.
        current_branch_id: Currently active branch ID.
    """
    if not branches:
        st.info("No branches yet. Start a conversation and use 'Branch from here' to create branches.")
        return
    
    st.markdown("### 🌿 Branches")
    
    for branch in branches:
        is_active = branch.id == current_branch_id
        
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                icon = "✅" if is_active else "📄"
                st.markdown(f"{icon} **{branch.name}**")
            
            with col2:
                st.caption(f"{branch.message_count} messages")
            
            with col3:
                st.caption(branch.created_at.strftime('%H:%M'))
            
            st.markdown("---")


def render_branch_actions(
    message_index: int,
    message_content: str,
    on_branch: callable = None
) -> None:
    """Render branch action button for a message.
    
    Args:
        message_index: Index of the message.
        message_content: Preview of message content.
        on_branch: Callback when branch is created.
    """
    col1, col2 = st.columns([3, 1])
    
    with col1:
        preview = message_content[:50] + "..." if len(message_content) > 50 else message_content
        st.caption(f"💬 {preview}")
    
    with col2:
        if st.button(
            "🔀 Branch",
            key=f"branch_{message_index}",
            help="Create a new branch from this message"
        ):
            if on_branch:
                on_branch(message_index)


def render_branch_manager(
    branches: list,
    current_branch_id: str = None,
    on_switch: callable = None,
    on_delete: callable = None
) -> None:
    """Render branch management UI.
    
    Args:
        branches: List of branches.
        current_branch_id: Currently active branch.
        on_switch: Callback when branch is switched.
        on_delete: Callback when branch is deleted.
    """
    st.markdown("### 🌿 Branch Manager")
    
    selected_branch = render_branch_dropdown(branches, current_branch_id)
    
    if selected_branch != current_branch_id and selected_branch is not None:
        if on_switch:
            on_switch(selected_branch)
            st.rerun()
    
    if branches:
        st.markdown("")
        
        with st.expander("🗑️ Manage Branches"):
            for branch in branches:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    is_active = branch.id == current_branch_id
                    status = "✅" if is_active else "📄"
                    st.write(f"{status} {branch.name}")
                
                with col2:
                    st.caption(f"{branch.message_count} msgs")
                
                with col3:
                    if not is_active and on_delete:
                        if st.button("Delete", key=f"del_{branch.id}"):
                            on_delete(branch.id)
                            st.rerun()
