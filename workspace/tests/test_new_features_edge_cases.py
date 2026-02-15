"""Edge case tests for new features (TokenTracker, BranchManager, ExportService, SessionManager)."""

import pytest
from unittest.mock import MagicMock, patch
import json

from core.session import Message, Branch, SessionManager
from core.token_tracker import TokenTracker, TokenBreakdown, TokenUsage
from core.branch_manager import BranchManager
from core.export import ExportService


@pytest.mark.edge_case
class TestTokenTrackerEdgeCases:
    """Edge case tests for TokenTracker."""
    
    def test_count_tokens_unicode(self):
        """Test token counting with unicode characters."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            mock_tokenizer_instance.encode.return_value = [1, 2]
            
            tracker = TokenTracker()
            
            result = tracker.count_tokens("Hello 世界 🌍")
            
            assert result >= 0
    
    def test_count_tokens_very_long_text(self):
        """Test token counting with very long text."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            long_text = "word " * 10000
            
            tracker = TokenTracker()
            tracker._tokenizer = mock_tokenizer_instance
            
            try:
                mock_tokenizer_instance.encode.side_effect = Exception("Too long")
            except:
                pass
            
            result = tracker.count_tokens(long_text)
            
            assert result > 0
    
    def test_count_tokens_special_characters(self):
        """Test token counting with special characters."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            tracker = TokenTracker()
            
            special_text = "Hello <>&\"'{}[]|\n\t\r"
            
            result = tracker.count_tokens(special_text)
            
            assert result >= 0
    
    def test_context_breakdown_zero_max_tokens(self):
        """Test context breakdown with zero max tokens."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            tracker = TokenTracker()
            tracker._max_context_tokens = 0
            
            breakdown = tracker.get_context_breakdown(
                messages=[],
                system_prompt="System",
                context_docs=["Doc"]
            )
            
            assert breakdown.percentage == 0
    
    def test_estimate_response_tokens_zero_max(self):
        """Test estimate with zero max context."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            tracker = TokenTracker()
            tracker._max_context_tokens = 0
            
            available = tracker.estimate_response_tokens("Hello")
            
            assert available >= 0


@pytest.mark.edge_case
class TestBranchManagerEdgeCases:
    """Edge case tests for BranchManager."""
    
    def test_create_branch_from_last_message(self):
        """Test creating branch from last message."""
        session_manager = SessionManager()
        
        for i in range(5):
            session_manager.add_user_message(f"Message {i}")
        
        manager = BranchManager(session_manager)
        
        messages = session_manager.get_messages()
        branch = manager.create_branch(messages[-1])
        
        assert branch is not None
    
    def test_delete_all_branches(self):
        """Test deleting all branches."""
        session_manager = SessionManager()
        
        for i in range(3):
            session_manager.add_user_message(f"Message {i}")
        
        manager = BranchManager(session_manager)
        
        messages = session_manager.get_messages()
        
        for i in range(3):
            manager.create_branch(messages[0], name=f"Branch {i}")
        
        for branch in manager.list_branches():
            manager.delete_branch(branch.id)
        
        assert len(manager.list_branches()) == 0
    
    def test_merge_same_branch(self):
        """Test merging branch with itself."""
        session_manager = SessionManager()
        session_manager.add_user_message("Hello")
        
        manager = BranchManager(session_manager)
        
        messages = session_manager.get_messages()
        branch = manager.create_branch(messages[0])
        
        result = manager.merge_branch(branch.id, branch.id)
        
        assert result is False
    
    def test_switch_branch_after_delete(self):
        """Test switching to branch after current was deleted."""
        session_manager = SessionManager()
        
        session_manager.add_user_message("Hello")
        
        manager = BranchManager(session_manager)
        
        messages = session_manager.get_messages()
        branch1 = manager.create_branch(messages[0], name="Branch 1")
        branch2 = manager.create_branch(messages[0], name="Branch 2")
        
        manager.delete_branch(branch1.id)
        
        result = session_manager.switch_branch(branch2.id)
        
        assert result is True


@pytest.mark.edge_case
class TestExportServiceEdgeCases:
    """Edge case tests for ExportService."""
    
    def test_export_markdown_very_long_content(self):
        """Test exporting message with very long content."""
        service = ExportService()
        
        long_content = "A" * 10000
        messages = [Message(role="user", content=long_content)]
        
        result = service.export_markdown(messages)
        
        assert long_content[:100] in result
    
    def test_export_json_special_characters(self):
        """Test exporting JSON with special characters."""
        service = ExportService()
        
        messages = [
            Message(role="user", content="Hello <>&\"'{}[]|\n\t\r")
        ]
        
        result = service.export_json(messages, include_metadata=True)
        
        data = json.loads(result)
        
        assert "Hello" in data["messages"][0]["content"]
    
    def test_export_plain_text_unicode(self):
        """Test exporting plain text with unicode."""
        service = ExportService()
        
        messages = [
            Message(role="user", content="Hello 世界 🌍")
        ]
        
        result = service.export_plain_text(messages)
        
        assert "Hello" in result
    
    def test_export_markdown_many_messages(self):
        """Test exporting many messages."""
        service = ExportService()
        
        messages = [
            Message(role="user", content=f"Message {i}")
            for i in range(100)
        ]
        
        result = service.export_markdown(messages)
        
        assert "Message 0" in result
        assert "Message 99" in result
    
    def test_export_with_branch_info_empty_branches(self):
        """Test export with branch info but no branches."""
        service = ExportService()
        
        messages = [Message(role="user", content="Hello")]
        
        result = service.export_with_branch_info(messages, [])
        
        data = json.loads(result)
        
        assert data["branches"] == []


@pytest.mark.edge_case
class TestSessionManagerEdgeCases:
    """Edge case tests for SessionManager enhanced features."""
    
    def test_pin_all_messages(self):
        """Test pinning all messages."""
        manager = SessionManager()
        
        for i in range(10):
            msg = manager.add_user_message(f"Message {i}")
            manager.pin_message(msg.id)
        
        pinned_count = sum(1 for m in manager.get_messages() if m.is_pinned)
        
        assert pinned_count == 10
    
    def test_feedback_all_messages(self):
        """Test setting feedback on all messages."""
        manager = SessionManager()
        
        for i in range(5):
            msg = manager.add_assistant_message(f"Answer {i}")
            manager.set_feedback(msg.id, "positive" if i % 2 == 0 else "negative")
        
        positive = sum(1 for m in manager.get_messages() if m.feedback == "positive")
        negative = sum(1 for m in manager.get_messages() if m.feedback == "negative")
        
        assert positive == 3
        assert negative == 2
    
    def test_search_unicode_query(self):
        """Test searching with unicode query."""
        manager = SessionManager()
        
        manager.add_user_message("Hello 世界")
        
        results = manager.search_messages("世界")
        
        assert len(results) == 1
    
    def test_delete_all_messages(self):
        """Test deleting all messages."""
        manager = SessionManager()
        
        for i in range(10):
            msg = manager.add_user_message(f"Message {i}")
            manager.delete_message(msg.id)
        
        assert len(manager.get_messages()) == 0
    
    def test_branch_with_no_messages(self):
        """Test creating branch when no messages exist."""
        manager = SessionManager()
        
        branch_id = manager.create_branch(0)
        
        assert branch_id == ""
    
    def test_get_message_by_index_empty(self):
        """Test getting message by index on empty session."""
        manager = SessionManager()
        
        msg = manager.get_message_by_index(0)
        
        assert msg is None


@pytest.mark.error_handling
class TestTokenTrackerErrorHandling:
    """Error handling tests for TokenTracker."""
    
    def test_tokenizer_error_handling(self):
        """Test that tokenizer errors are handled gracefully."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            mock_tokenizer_instance.encode.side_effect = Exception("Tokenizer failed")
            
            tracker = TokenTracker()
            tracker._tokenizer = mock_tokenizer_instance
            
            result = tracker.count_tokens("Test")
            
            assert result > 0
    
    def test_load_tokenizer_failure(self):
        """Test tokenizer loading failure."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer.from_pretrained.side_effect = Exception("Load failed")
            
            tracker = TokenTracker()
            
            assert tracker._tokenizer is None


@pytest.mark.error_handling
class TestBranchManagerErrorHandling:
    """Error handling tests for BranchManager."""
    
    def test_create_branch_none_message(self):
        """Test creating branch with None message."""
        session_manager = SessionManager()
        
        manager = BranchManager(session_manager)
        
        branch = manager.create_branch(None)
        
        assert branch is None
    
    def test_get_branch_none_id(self):
        """Test getting branch with None ID."""
        session_manager = SessionManager()
        
        manager = BranchManager(session_manager)
        
        branch = manager.get_branch(None)
        
        assert branch is None


@pytest.mark.error_handling
class TestExportServiceErrorHandling:
    """Error handling tests for ExportService."""
    
    def test_export_with_malformed_message(self):
        """Test exporting with malformed message data."""
        service = ExportService()
        
        msg = Message(
            role="user",
            content="Test",
            timestamp=None
        )
        
        result = service.export_markdown([msg])
        
        assert "Test" in result
