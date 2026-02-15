"""Unit tests for model.prompts module."""

import pytest
from model.prompts import (
    build_rag_prompt,
    build_reasoning_prompt,
    PromptBuilder,
)


@pytest.mark.unit
class TestBuildRAGPrompt:
    """Test build_rag_prompt function."""
    
    def test_basic_prompt(self):
        """Test basic prompt building."""
        query = "What is AI?"
        context = "AI is artificial intelligence."
        
        prompt = build_rag_prompt(query, context)
        
        assert "System:" in prompt
        assert query in prompt
        assert context in prompt
        assert "Context:" in prompt
        assert "Assistant:" in prompt
    
    def test_prompt_with_history(self):
        """Test prompt with conversation history."""
        query = "Tell me more"
        context = "Context here"
        history = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
        ]
        
        prompt = build_rag_prompt(query, context, history)
        
        assert "What is AI?" in prompt
        assert "AI is artificial intelligence." in prompt
        assert "Conversation History:" in prompt
    
    def test_prompt_without_context(self):
        """Test prompt without context."""
        query = "Hello"
        
        prompt = build_rag_prompt(query, "")
        
        assert query in prompt
        assert "System:" in prompt
    
    def test_prompt_custom_system(self):
        """Test prompt with custom system prompt."""
        query = "Test"
        context = "Context"
        custom_system = "You are a custom assistant."
        
        prompt = build_rag_prompt(query, context, system_prompt=custom_system)
        
        assert custom_system in prompt
    
    def test_prompt_history_limit(self):
        """Test that history is limited to MAX_CHAT_HISTORY."""
        query = "Latest question"
        context = "Context"
        # Create many history entries
        history = [
            {"role": "user", "content": f"Question {i}"}
            for i in range(20)
        ]
        
        prompt = build_rag_prompt(query, context, history)
        
        # Should only include last 10 messages (default MAX_CHAT_HISTORY)
        assert prompt.count("Question") <= 10


@pytest.mark.unit
class TestBuildReasoningPrompt:
    """Test build_reasoning_prompt function."""
    
    def test_reasoning_prompt_structure(self):
        """Test reasoning prompt structure."""
        query = "Explain quantum computing"
        context = "Quantum computing uses qubits."
        
        prompt = build_reasoning_prompt(query, context)
        
        assert "Context information:" in prompt
        assert context in prompt
        assert "Question:" in prompt
        assert query in prompt
        assert "REASONING:" in prompt
        assert "ANSWER:" in prompt
        assert "Instructions:" in prompt
    
    def test_reasoning_prompt_instructions(self):
        """Test that reasoning prompt includes instructions."""
        prompt = build_reasoning_prompt("Test", "Context")
        
        assert "step-by-step" in prompt.lower()
        assert "analyze" in prompt.lower()
        assert "Format your response" in prompt


@pytest.mark.unit
class TestPromptBuilder:
    """Test PromptBuilder class."""
    
    def test_chat_prompt(self):
        """Test PromptBuilder.chat_prompt."""
        query = "Hello"
        context = "World"
        
        prompt = PromptBuilder.chat_prompt(query, context)
        
        assert query in prompt
        assert context in prompt
    
    def test_reasoning_prompt_static(self):
        """Test PromptBuilder.reasoning_prompt."""
        query = "Test"
        context = "Context"
        
        prompt = PromptBuilder.reasoning_prompt(query, context)
        
        assert query in prompt
        assert context in prompt
        assert "REASONING:" in prompt
    
    def test_summarize_prompt(self):
        """Test PromptBuilder.summarize_prompt."""
        text = "This is a long text to summarize."
        max_length = 100
        
        prompt = PromptBuilder.summarize_prompt(text, max_length)
        
        assert "summarize" in prompt.lower()
        assert text in prompt
        assert str(max_length) in prompt
        assert "Summary:" in prompt
    
    def test_summarize_prompt_default_length(self):
        """Test summarize prompt with default length."""
        text = "Text to summarize"
        
        prompt = PromptBuilder.summarize_prompt(text)
        
        assert "200" in prompt  # Default max_length
