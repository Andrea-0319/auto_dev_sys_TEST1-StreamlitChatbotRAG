"""Prompt templates and builders."""

from typing import List, Optional

from config import settings


def build_rag_prompt(
    query: str,
    context: str,
    history: Optional[List[dict]] = None,
    system_prompt: Optional[str] = None
) -> str:
    """Build a RAG prompt with context and history.
    
    Args:
        query: User query.
        context: Retrieved context from documents.
        history: Optional conversation history.
        system_prompt: Optional custom system prompt.
        
    Returns:
        Formatted prompt string.
    """
    if system_prompt is None:
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
Follow these guidelines:
1. Use the provided context to answer the question accurately
2. If the context doesn't contain the answer, say so clearly
3. Show your reasoning process step by step
4. Be concise but thorough in your explanations
5. Cite the source documents when providing information"""
    
    # Build conversation history
    history_str = ""
    if history:
        for msg in history[-settings.MAX_CHAT_HISTORY:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                history_str += f"User: {content}\n"
            else:
                history_str += f"Assistant: {content}\n"
    
    # Build the full prompt
    prompt_parts = [f"System: {system_prompt}"]
    
    if context:
        prompt_parts.append(f"\nContext:\n{context}")
    
    if history_str:
        prompt_parts.append(f"\nConversation History:\n{history_str}")
    
    prompt_parts.append(f"\nUser: {query}")
    prompt_parts.append("\nAssistant: Let me think through this step by step.")
    
    return "\n".join(prompt_parts)


def build_reasoning_prompt(query: str, context: str) -> str:
    """Build a prompt that encourages explicit reasoning.
    
    Args:
        query: User query.
        context: Retrieved context.
        
    Returns:
        Formatted prompt string.
    """
    return f"""You are an AI assistant that provides detailed reasoning before answering.

Context information:
{context}

Question: {query}

Instructions:
1. First, analyze the context and identify relevant information
2. Show your step-by-step reasoning process
3. Provide a clear, well-structured answer
4. Format your response with clear sections for reasoning and answer

Format your response like this:
REASONING:
[Your step-by-step analysis and thinking process]

ANSWER:
[Your final answer based on the reasoning above]"""


class PromptBuilder:
    """Build prompts for different scenarios."""
    
    @staticmethod
    def chat_prompt(query: str, context: str = "", history: Optional[List[dict]] = None) -> str:
        """Build a standard chat prompt."""
        return build_rag_prompt(query, context, history)
    
    @staticmethod
    def reasoning_prompt(query: str, context: str = "") -> str:
        """Build a prompt that encourages reasoning."""
        return build_reasoning_prompt(query, context)
    
    @staticmethod
    def summarize_prompt(text: str, max_length: int = 200) -> str:
        """Build a summarization prompt."""
        return f"""Please summarize the following text in {max_length} words or less:

{text}

Summary:"""
