"""Unit tests for core.summarization module."""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from core.summarization import SummarizationService, SummaryResult, get_summarization_service
from core.session import Message


@pytest.mark.unit
class TestSummaryResult:
    """Test SummaryResult dataclass."""
    
    def test_summary_result_creation(self):
        """Test SummaryResult creation."""
        msg = Message(role="system", content="Summary content")
        result = SummaryResult(
            summary="This is a summary",
            original_message_ids=["msg-1", "msg-2"],
            new_message=msg,
            tokens_saved=100
        )
        
        assert result.summary == "This is a summary"
        assert result.original_message_ids == ["msg-1", "msg-2"]
        assert result.new_message == msg
        assert result.tokens_saved == 100


@pytest.mark.unit
class TestSummarizationServiceInitialization:
    """Test SummarizationService initialization."""
    
    def test_initialization(self):
        """Test SummarizationService initialization."""
        mock_handler = Mock()
        service = SummarizationService(mock_handler)
        
        assert service._model_handler is mock_handler


@pytest.mark.unit
class TestSummarizationServiceSummarize:
    """Test message summarization."""
    
    def test_summarize_empty_messages(self):
        """Test summarizing empty messages."""
        mock_handler = Mock()
        service = SummarizationService(mock_handler)
        
        result = service.summarize_messages([])
        
        assert result.summary == ""
        assert result.original_message_ids == []
        assert result.new_message is None
        assert result.tokens_saved == 0
    
    def test_summarize_fewer_than_preserve(self):
        """Test summarizing fewer messages than preserve count."""
        mock_handler = Mock()
        service = SummarizationService(mock_handler)
        
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there")
        ]
        
        result = service.summarize_messages(messages, preserve_recent=4)
        
        assert result.summary == ""
        assert result.original_message_ids == []
        assert result.new_message is None
        assert result.tokens_saved == 0
    
    def test_summarize_messages_success(self):
        """Test successful message summarization."""
        mock_handler = Mock()
        mock_handler.generate.return_value = {"response": "This is a concise summary."}
        
        service = SummarizationService(mock_handler)
        
        messages = [
            Message(role="user", content="What is AI?"),
            Message(role="assistant", content="AI is artificial intelligence."),
            Message(role="user", content="What is ML?"),
            Message(role="assistant", content="ML is machine learning."),
            Message(role="user", content="What is NLP?"),
            Message(role="assistant", content="NLP is natural language processing."),
            Message(role="user", content="Tell me more."),
            Message(role="assistant", content="Sure, here's more.")
        ]
        
        result = service.summarize_messages(messages, preserve_recent=2)
        
        assert result.summary == "This is a concise summary."
        assert len(result.original_message_ids) == 6
        assert result.new_message is not None
        assert result.tokens_saved >= 0
        mock_handler.generate.assert_called_once()
    
    def test_summarize_with_reasoning(self):
        """Test summarizing messages with reasoning."""
        mock_handler = Mock()
        mock_handler.generate.return_value = {"response": "Summary with reasoning."}
        
        service = SummarizationService(mock_handler)
        
        messages = [
            Message(
                role="user", 
                content="Question?",
                reasoning="Thinking about the question"
            ),
            Message(
                role="assistant", 
                content="Answer",
                reasoning="Thinking about the answer"
            ),
            Message(role="user", content="Another question?"),
            Message(role="assistant", content="Another answer."),
            Message(role="user", content="Final question?"),
            Message(role="assistant", content="Final answer.")
        ]
        
        result = service.summarize_messages(messages, preserve_recent=2)
        
        assert result.summary == "Summary with reasoning."
        assert len(result.original_message_ids) == 4
    
    def test_summarize_generation_failure(self):
        """Test handling generation failure."""
        mock_handler = Mock()
        mock_handler.generate.side_effect = Exception("Generation failed")
        
        service = SummarizationService(mock_handler)
        
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
            Message(role="user", content="How are you?"),
            Message(role="assistant", content="I'm good."),
            Message(role="user", content="What's up?"),
            Message(role="assistant", content="Nothing much.")
        ]
        
        result = service.summarize_messages(messages)
        
        assert result.summary == ""
        assert result.original_message_ids == []
        assert result.new_message is None
        assert result.tokens_saved == 0
    
    def test_summarize_preserve_recent_default(self):
        """Test default preserve_recent value."""
        mock_handler = Mock()
        mock_handler.generate.return_value = {"response": "Summary"}
        
        service = SummarizationService(mock_handler)
        
        messages = [
            Message(role="user", content=f"Message {i}") 
            for i in range(10)
        ]
        
        result = service.summarize_messages(messages)
        
        assert len(result.original_message_ids) == 6
        assert result.new_message is not None


@pytest.mark.unit
class TestSummarizationServiceCreateMessage:
    """Test summary message creation."""
    
    def test_create_summary_message(self):
        """Test creating summary message."""
        mock_handler = Mock()
        service = SummarizationService(mock_handler)
        
        msg = service.create_summary_message(
            "This is the summary",
            ["msg-1", "msg-2", "msg-3"]
        )
        
        assert msg.role == "system"
        assert "This is the summary" in msg.content
        assert "summarized:msg-1" in msg.sources
        assert "summarized:msg-2" in msg.sources
        assert "summarized:msg-3" in msg.sources
    
    def test_create_summary_message_empty_ids(self):
        """Test creating summary with no original IDs."""
        mock_handler = Mock()
        service = SummarizationService(mock_handler)
        
        msg = service.create_summary_message("Summary", [])
        
        assert msg.role == "system"
        assert "Summary" in msg.content


@pytest.mark.unit
class TestSummarizationServicePrompt:
    """Test summary prompt generation."""
    
    def test_get_summary_prompt_empty(self):
        """Test prompt with empty messages."""
        mock_handler = Mock()
        service = SummarizationService(mock_handler)
        
        prompt = service.get_summary_prompt([])
        
        assert "Please provide a concise summary" in prompt
        assert "CONVERSATION:" in prompt
        assert "SUMMARY" in prompt
    
    def test_get_summary_prompt_with_messages(self):
        """Test prompt with messages."""
        mock_handler = Mock()
        service = SummarizationService(mock_handler)
        
        messages = [
            Message(role="user", content="What is AI?"),
            Message(role="assistant", content="AI is artificial intelligence.")
        ]
        
        prompt = service.get_summary_prompt(messages)
        
        assert "What is AI?" in prompt
        assert "AI is artificial intelligence" in prompt
        assert "User:" in prompt
        assert "Assistant:" in prompt
    
    def test_get_summary_prompt_format(self):
        """Test prompt format."""
        mock_handler = Mock()
        service = SummarizationService(mock_handler)
        
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
            Message(role="user", content="How are you?")
        ]
        
        prompt = service.get_summary_prompt(messages)
        
        lines = prompt.split("\n")
        assert any("User:" in line for line in lines)
        assert any("Assistant:" in line for line in lines)


@pytest.mark.unit
class TestGetSummarizationService:
    """Test get_summarization_service function."""
    
    def test_get_summarization_service_with_handler(self):
        """Test getting service with handler."""
        import core.summarization as sum_module
        sum_module._summarization_service = None
        
        mock_handler = Mock()
        service = get_summarization_service(mock_handler)
        
        assert service is not None
        assert service._model_handler is mock_handler
    
    def test_get_summarization_service_singleton(self):
        """Test singleton behavior."""
        import core.summarization as sum_module
        sum_module._summarization_service = None
        
        mock_handler = Mock()
        
        service1 = get_summarization_service(mock_handler)
        service2 = get_summarization_service()
        
        assert service1 is service2
    
    def test_get_summarization_service_no_handler(self):
        """Test getting service without handler."""
        import core.summarization as sum_module
        sum_module._summarization_service = None
        
        service = get_summarization_service(None)
        
        assert service is None


@pytest.mark.edge_case
class TestSummarizationEdgeCases:
    """Edge case tests for summarization."""
    
    def test_summarize_very_long_messages(self):
        """Test summarizing very long messages."""
        mock_handler = Mock()
        mock_handler.generate.return_value = {"response": "Short summary."}
        
        service = SummarizationService(mock_handler)
        
        long_content = "word " * 1000
        messages = [
            Message(role="user", content=long_content),
            Message(role="assistant", content=long_content),
            Message(role="user", content=long_content),
            Message(role="user", content="Question?"),
            Message(role="assistant", content="Answer.")
        ]
        
        result = service.summarize_messages(messages)
        
        assert result.summary == "Short summary."
        assert result.tokens_saved > 0
    
    def test_summarize_unicode_content(self):
        """Test summarizing unicode content."""
        mock_handler = Mock()
        mock_handler.generate.return_value = {"response": "Summary with unicode."}
        
        service = SummarizationService(mock_handler)
        
        messages = [
            Message(role="user", content="Hello 世界 🌍"),
            Message(role="assistant", content="Greetings 日本語"),
            Message(role="user", content="Question 1?"),
            Message(role="assistant", content="Answer 1."),
            Message(role="user", content="Question 2?"),
            Message(role="assistant", content="Answer 2."),
            Message(role="user", content="Keep this"),
            Message(role="assistant", content="Keeping this too")
        ]
        
        result = service.summarize_messages(messages, preserve_recent=2)
        
        assert len(result.original_message_ids) > 0
    
    def test_summarize_preserve_single_message(self):
        """Test with preserve_recent=1."""
        mock_handler = Mock()
        mock_handler.generate.return_value = {"response": "Summary"}
        
        service = SummarizationService(mock_handler)
        
        messages = [
            Message(role="user", content="Msg1"),
            Message(role="assistant", content="Msg2"),
            Message(role="user", content="Msg3"),
            Message(role="assistant", content="Msg4"),
            Message(role="user", content="Msg5"),
            Message(role="assistant", content="Msg6"),
            Message(role="user", content="Keep this")
        ]
        
        result = service.summarize_messages(messages, preserve_recent=1)
        
        assert len(result.original_message_ids) == 6
    
    def test_summarize_preserve_zero(self):
        """Test with preserve_recent=0."""
        mock_handler = Mock()
        mock_handler.generate.return_value = {"response": "Summary"}
        
        service = SummarizationService(mock_handler)
        
        messages = [
            Message(role="user", content="Msg1"),
            Message(role="assistant", content="Msg2"),
            Message(role="user", content="Msg3")
        ]
        
        result = service.summarize_messages(messages, preserve_recent=0)
        
        assert len(result.original_message_ids) == 3


@pytest.mark.error_handling
class TestSummarizationErrorHandling:
    """Error handling tests for summarization."""
    
    def test_generate_returns_non_dict(self):
        """Test handling non-dict response."""
        mock_handler = Mock()
        mock_handler.generate.return_value = "Just a string response"
        
        service = SummarizationService(mock_handler)
        
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
            Message(role="user", content="How are you?"),
            Message(role="assistant", content="Good")
        ]
        
        result = service.summarize_messages(messages)
        
        assert result.summary == "Just a string response"
    
    def test_generate_response_without_response_key(self):
        """Test handling response without response key."""
        mock_handler = Mock()
        mock_handler.generate.return_value = {"answer": "Response without response key"}
        
        service = SummarizationService(mock_handler)
        
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
            Message(role="user", content="How are you?"),
            Message(role="assistant", content="Good")
        ]
        
        result = service.summarize_messages(messages)
        
        assert result.summary == ""
    
    def test_create_message_with_none_content(self):
        """Test creating summary with None content."""
        mock_handler = Mock()
        service = SummarizationService(mock_handler)
        
        msg = service.create_summary_message(None, ["msg-1"])
        
        assert msg.role == "system"
        assert msg.content is not None
