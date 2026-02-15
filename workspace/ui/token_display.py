"""Token display UI components."""

import streamlit as st

from core.token_tracker import TokenUsage, TokenBreakdown


def render_token_indicator(usage: TokenUsage) -> None:
    """Render token usage indicator.
    
    Args:
        usage: Current token usage.
    """
    percentage = usage.percentage
    
    if percentage >= 90:
        color = "red"
        icon = "🔴"
    elif percentage >= 80:
        color = "orange"
        icon = "🟠"
    else:
        color = "green"
        icon = "🟢"
    
    st.markdown(f"""
    <div style="
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
    ">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 20px;">{icon}</span>
            <div style="flex: 1;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span><strong>Context Usage:</strong></span>
                    <span>{usage.current} / {usage.max} tokens ({percentage:.1f}%)</span>
                </div>
                <div style="
                    height: 8px;
                    background-color: #e0e0e0;
                    border-radius: 4px;
                    overflow: hidden;
                ">
                    <div style="
                        height: 100%;
                        width: {min(percentage, 100)}%;
                        background-color: {color};
                        border-radius: 4px;
                        transition: width 0.3s ease;
                    "></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_token_breakdown(breakdown: TokenBreakdown) -> None:
    """Render detailed token breakdown.
    
    Args:
        breakdown: Token breakdown by category.
    """
    with st.expander("📊 Token Breakdown", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "System Prompt",
                f"{breakdown.system_prompt_tokens}",
                help="Tokens used by system prompt"
            )
        
        with col2:
            st.metric(
                "Chat History",
                f"{breakdown.chat_history_tokens}",
                help="Tokens used by conversation history"
            )
        
        with col3:
            st.metric(
                "Retrieved Context",
                f"{breakdown.context_tokens}",
                help="Tokens used by retrieved document context"
            )
        
        st.markdown("---")
        
        total_col1, total_col2 = st.columns(2)
        
        with total_col1:
            st.metric("Total Tokens", f"{breakdown.total}")
        
        with total_col2:
            usage_percent = breakdown.percentage
            if usage_percent >= 80:
                status = "⚠️ High"
            else:
                status = "✅ Normal"
            st.metric("Usage Status", status)


def render_token_warning(usage: TokenUsage) -> None:
    """Render warning message if approaching limit.
    
    Args:
        usage: Current token usage.
    """
    if usage.percentage >= 80:
        if usage.percentage >= 90:
            st.error(
                f"⚠️ **Critical**: Context window is {usage.percentage:.1f}% full! "
                "Consider summarizing the conversation or starting a new one."
            )
        elif usage.percentage >= 80:
            st.warning(
                f"⚠️ **Warning**: Context window is {usage.percentage:.1f}% full. "
                "You may want to summarize the conversation soon."
            )


def render_token_display_full(
    usage: TokenUsage,
    show_breakdown: bool = True,
    show_warning: bool = True
) -> None:
    """Render complete token display with indicator, breakdown, and warning.
    
    Args:
        usage: Current token usage.
        show_breakdown: Whether to show breakdown.
        show_warning: Whether to show warning.
    """
    render_token_indicator(usage)
    
    if show_warning:
        render_token_warning(usage)
    
    if show_breakdown:
        render_token_breakdown(usage.breakdown)
