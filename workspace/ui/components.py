"""Reusable UI components."""

import streamlit as st
from typing import Optional

from config import settings


def render_header():
    """Render application header."""
    st.title(f"{settings.PAGE_ICON} {settings.PAGE_TITLE}")
    st.markdown("---")


def render_footer():
    """Render application footer."""
    st.markdown("---")
    st.caption("Built with Streamlit | Powered by Hugging Face")


def render_info_box(message: str, box_type: str = "info"):
    """Render an info box.
    
    Args:
        message: Message to display.
        box_type: Type of box (info, success, warning, error).
    """
    if box_type == "info":
        st.info(message)
    elif box_type == "success":
        st.success(message)
    elif box_type == "warning":
        st.warning(message)
    elif box_type == "error":
        st.error(message)


def render_spinner(text: str = "Processing..."):
    """Render a spinner with text.
    
    Args:
        text: Text to display.
        
    Returns:
        Spinner context manager.
    """
    return st.spinner(text)


def render_progress_bar(progress: float, text: Optional[str] = None):
    """Render a progress bar.
    
    Args:
        progress: Progress value between 0 and 1.
        text: Optional text to display.
    """
    if text:
        st.text(text)
    st.progress(progress)


def render_expander(title: str, content: str, expanded: bool = False):
    """Render an expandable section.
    
    Args:
        title: Expander title.
        content: Content to display.
        expanded: Whether expanded by default.
    """
    with st.expander(title, expanded=expanded):
        st.markdown(content)


def render_chat_message(role: str, content: str, avatar: Optional[str] = None):
    """Render a chat message.
    
    Args:
        role: Message role (user/assistant).
        content: Message content.
        avatar: Optional avatar emoji.
    """
    if role == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar=avatar or "🤖"):
            st.markdown(content)


def render_badge(text: str, color: str = "blue"):
    """Render a badge.
    
    Args:
        text: Badge text.
        color: Badge color.
    """
    color_map = {
        "blue": "#1f77b4",
        "green": "#2ca02c",
        "red": "#d62728",
        "orange": "#ff7f0e",
        "gray": "#7f7f7f",
    }
    hex_color = color_map.get(color, color)
    st.markdown(
        f'<span style="background-color: {hex_color}; color: white; padding: 2px 8px; '
        f'border-radius: 10px; font-size: 0.8em;">{text}</span>',
        unsafe_allow_html=True
    )


def render_divider():
    """Render a horizontal divider."""
    st.markdown("---")


def render_empty_state(message: str = "No data available", icon: str = "📭"):
    """Render an empty state message.
    
    Args:
        message: Message to display.
        icon: Icon to show.
    """
    st.markdown(
        f'<div style="text-align: center; padding: 2rem; color: #888;">'
        f'<div style="font-size: 3rem;">{icon}</div>'
        f'<div>{message}</div>'
        f'</div>',
        unsafe_allow_html=True
    )
