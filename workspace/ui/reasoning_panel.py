"""Reasoning visualization panel."""

from typing import List, Optional

import streamlit as st


def render_reasoning_panel(reasoning: str, sources: Optional[List[str]] = None):
    """Render the reasoning panel.
    
    Args:
        reasoning: Reasoning text to display.
        sources: Optional list of source documents.
    """
    with st.container():
        st.markdown("### 💭 Model Reasoning")
        
        # Display reasoning steps
        if reasoning:
            steps = parse_reasoning_steps(reasoning)
            
            for i, step in enumerate(steps, 1):
                with st.expander(f"Step {i}", expanded=i == 1):
                    st.markdown(step)
        else:
            st.info("No detailed reasoning available for this response.")
        
        # Display sources
        if sources:
            st.markdown("#### 📚 Sources")
            for source in sources:
                st.markdown(f"- {source}")


def parse_reasoning_steps(reasoning: str) -> List[str]:
    """Parse reasoning text into steps.
    
    Args:
        reasoning: Raw reasoning text.
        
    Returns:
        List of reasoning steps.
    """
    if not reasoning:
        return []
    
    # Split by common step markers
    import re
    steps = re.split(r'\n+(?:Step \d+[.:]|\d+\.|•|\-)', reasoning)
    
    # Clean up steps
    steps = [step.strip() for step in steps if step.strip()]
    
    # If no clear steps, split by sentences
    if len(steps) <= 1:
        steps = [s.strip() for s in reasoning.split('.') if s.strip()]
    
    return steps if steps else [reasoning]


def render_retrieval_info(chunks: List[dict]):
    """Render retrieved chunks information.
    
    Args:
        chunks: List of retrieved chunks.
    """
    with st.expander("🔍 Retrieved Context", expanded=False):
        if not chunks:
            st.info("No relevant context found.")
            return
        
        st.markdown(f"**Found {len(chunks)} relevant chunks:**")
        
        for i, chunk in enumerate(chunks, 1):
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    doc_name = chunk.get("document_name", "Unknown")
                    text = chunk.get("text", "")
                    score = chunk.get("score", 0)
                    
                    st.markdown(f"**Chunk {i}** from *{doc_name}*")
                    st.markdown(f"```\n{text[:300]}{'...' if len(text) > 300 else ''}\n```")
                
                with col2:
                    st.metric("Relevance", f"{score:.3f}")
            
            if i < len(chunks):
                st.divider()


def render_thinking_animation():
    """Render an animated thinking indicator."""
    import time
    
    placeholder = st.empty()
    phrases = [
        "Analyzing the question...",
        "Searching knowledge base...",
        "Formulating reasoning steps...",
        "Synthesizing information...",
    ]
    
    for phrase in phrases:
        placeholder.markdown(f"💭 {phrase}")
        time.sleep(0.3)
    
    placeholder.empty()


def render_confidence_indicator(confidence: Optional[float]):
    """Render confidence indicator.
    
    Args:
        confidence: Confidence score between 0 and 1.
    """
    if confidence is None:
        return
    
    if confidence >= 0.8:
        color = "green"
        label = "High"
    elif confidence >= 0.5:
        color = "orange"
        label = "Medium"
    else:
        color = "red"
        label = "Low"
    
    st.markdown(
        f'<span style="color: {color}; font-weight: bold;">'
        f'Confidence: {label} ({confidence:.0%})</span>',
        unsafe_allow_html=True
    )
