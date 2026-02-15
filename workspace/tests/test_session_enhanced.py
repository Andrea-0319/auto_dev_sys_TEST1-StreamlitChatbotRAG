"""Unit tests for enhanced SessionManager features (branching, pinning, feedback, search)."""

import pytest
from datetime import datetime

from core.session import Message, Branch, SessionManager


@pytest.mark.unit
class TestSessionManagerBranching:
    """Test SessionManager branching functionality."""
    
    def test_create_branch_basic(self):
        """Test creating a basic branch."""
        manager = SessionManager()
        
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi there")
        
        branch_id = manager.create_branch(0, "Test Branch")
        
        assert branch_id != ""
        assert manager.get_current_branch_id() == branch_id
    
    def test_create_branch_invalid_index(self):
        """Test creating branch with invalid index."""
        manager = SessionManager()
        
        manager.add_user_message("Hello")
        
        branch_id = manager.create_branch(10)
        
        assert branch_id == ""
    
    def test_create_branch_negative_index(self):
        """Test creating branch with negative index."""
        manager = SessionManager()
        
        branch_id = manager.create_branch(-1)
        
        assert branch_id == ""
    
    def test_switch_branch(self):
        """Test switching to a different branch."""
        manager = SessionManager()
        
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")
        
        branch1_id = manager.create_branch(0, "Branch 1")
        branch2_id = manager.create_branch(0, "Branch 2")
        
        result = manager.switch_branch(branch1_id)
        
        assert result is True
        assert manager.get_current_branch_id() == branch1_id
    
    def test_switch_branch_not_exists(self):
        """Test switching to non-existent branch."""
        manager = SessionManager()
        
        result = manager.switch_branch("non-existent-id")
        
        assert result is False
    
    def test_get_current_branch_id(self):
        """Test getting current branch ID."""
        manager = SessionManager()
        
        assert manager.get_current_branch_id() is None
        
        manager.add_user_message("Hello")
        branch_id = manager.create_branch(0)
        
        assert manager.get_current_branch_id() == branch_id
    
    def test_get_all_branches(self):
        """Test getting all branches."""
        manager = SessionManager()
        
        manager.add_user_message("Hello")
        
        manager.create_branch(0, "Branch 1")
        manager.create_branch(0, "Branch 2")
        
        branches = manager.get_all_branches()
        
        assert len(branches) == 2
        assert branches[0].name == "Branch 1"
        assert branches[1].name == "Branch 2"
    
    def test_delete_branch(self):
        """Test deleting a branch."""
        manager = SessionManager()
        
        manager.add_user_message("Hello")
        
        branch_id = manager.create_branch(0)
        
        result = manager.delete_branch(branch_id)
        
        assert result is True
        assert branch_id not in [b.id for b in manager.get_all_branches()]
    
    def test_delete_branch_not_exists(self):
        """Test deleting non-existent branch."""
        manager = SessionManager()
        
        result = manager.delete_branch("non-existent")
        
        assert result is False
    
    def test_delete_current_branch(self):
        """Test deleting current branch."""
        manager = SessionManager()
        
        manager.add_user_message("Hello")
        branch_id = manager.create_branch(0)
        
        result = manager.delete_branch(branch_id)
        
        assert result is True
        assert manager.get_current_branch_id() is None


@pytest.mark.unit
class TestSessionManagerMessageMetadata:
    """Test SessionManager message metadata (pin, feedback)."""
    
    def test_pin_message(self):
        """Test pinning a message."""
        manager = SessionManager()
        
        msg = manager.add_user_message("Important message")
        
        result = manager.pin_message(msg.id)
        
        assert result is True
        assert msg.is_pinned is True
    
    def test_pin_message_not_exists(self):
        """Test pinning non-existent message."""
        manager = SessionManager()
        
        result = manager.pin_message("non-existent")
        
        assert result is False
    
    def test_unpin_message(self):
        """Test unpinning a message."""
        manager = SessionManager()
        
        msg = manager.add_user_message("Message")
        manager.pin_message(msg.id)
        
        result = manager.unpin_message(msg.id)
        
        assert result is True
        assert msg.is_pinned is False
    
    def test_unpin_message_not_exists(self):
        """Test unpinning non-existent message."""
        manager = SessionManager()
        
        result = manager.unpin_message("non-existent")
        
        assert result is False
    
    def test_set_feedback_positive(self):
        """Test setting positive feedback."""
        manager = SessionManager()
        
        msg = manager.add_assistant_message("Answer")
        
        result = manager.set_feedback(msg.id, "positive")
        
        assert result is True
        assert msg.feedback == "positive"
    
    def test_set_feedback_negative(self):
        """Test setting negative feedback."""
        manager = SessionManager()
        
        msg = manager.add_assistant_message("Answer")
        
        result = manager.set_feedback(msg.id, "negative")
        
        assert result is True
        assert msg.feedback == "negative"
    
    def test_set_feedback_not_exists(self):
        """Test setting feedback on non-existent message."""
        manager = SessionManager()
        
        result = manager.set_feedback("non-existent", "positive")
        
        assert result is False
    
    def test_delete_message(self):
        """Test deleting a message."""
        manager = SessionManager()
        
        msg = manager.add_user_message("To delete")
        
        result = manager.delete_message(msg.id)
        
        assert result is True
        assert msg not in manager.get_messages()
    
    def test_delete_message_not_exists(self):
        """Test deleting non-existent message."""
        manager = SessionManager()
        
        result = manager.delete_message("non-existent")
        
        assert result is False
    
    def test_message_metadata_persists(self):
        """Test that message metadata persists in exported data."""
        manager = SessionManager()
        
        msg = manager.add_assistant_message("Answer", reasoning="Think")
        manager.pin_message(msg.id)
        manager.set_feedback(msg.id, "positive")
        
        exported = manager.export_conversation()
        
        assert len(exported["messages"]) == 1
        assert exported["messages"][0]["is_pinned"] is True


@pytest.mark.unit
class TestSessionManagerSearch:
    """Test SessionManager search functionality."""
    
    def test_search_messages_empty(self):
        """Test searching in empty conversation."""
        manager = SessionManager()
        
        results = manager.search_messages("hello")
        
        assert results == []
    
    def test_search_messages_found(self):
        """Test searching and finding messages."""
        manager = SessionManager()
        
        manager.add_user_message("Hello world")
        manager.add_assistant_message("Goodbye world")
        manager.add_user_message("Another message")
        
        results = manager.search_messages("world")
        
        assert len(results) == 2
    
    def test_search_messages_case_insensitive(self):
        """Test search is case insensitive."""
        manager = SessionManager()
        
        manager.add_user_message("Hello World")
        
        results = manager.search_messages("hello")
        
        assert len(results) == 1
    
    def test_search_messages_not_found(self):
        """Test searching and not finding messages."""
        manager = SessionManager()
        
        manager.add_user_message("Hello")
        
        results = manager.search_messages("xyz")
        
        assert results == []
    
    def test_search_messages_partial_match(self):
        """Test partial matching in search."""
        manager = SessionManager()
        
        manager.add_user_message("The quick brown fox")
        
        results = manager.search_messages("quick")
        
        assert len(results) == 1
    
    def test_search_messages_empty_query(self):
        """Test search with empty query."""
        manager = SessionManager()
        
        manager.add_user_message("Hello")
        
        results = manager.search_messages("")
        
        assert len(results) == 1


@pytest.mark.unit
class TestSessionManagerGetMessageByIndex:
    """Test getting message by index."""
    
    def test_get_message_by_index_valid(self):
        """Test getting message by valid index."""
        manager = SessionManager()
        
        manager.add_user_message("First")
        manager.add_user_message("Second")
        
        msg = manager.get_message_by_index(1)
        
        assert msg is not None
        assert msg.content == "Second"
    
    def test_get_message_by_index_negative(self):
        """Test getting message with negative index."""
        manager = SessionManager()
        
        manager.add_user_message("First")
        
        msg = manager.get_message_by_index(-1)
        
        assert msg is None
    
    def test_get_message_by_index_out_of_bounds(self):
        """Test getting message with out of bounds index."""
        manager = SessionManager()
        
        manager.add_user_message("First")
        
        msg = manager.get_message_by_index(5)
        
        assert msg is None


@pytest.mark.unit
class TestMessageDataclass:
    """Test enhanced Message dataclass."""
    
    def test_message_with_pinned(self):
        """Test message with pinned flag."""
        msg = Message(role="user", content="Test", is_pinned=True)
        
        assert msg.is_pinned is True
    
    def test_message_with_feedback_positive(self):
        """Test message with positive feedback."""
        msg = Message(role="assistant", content="Answer", feedback="positive")
        
        assert msg.feedback == "positive"
    
    def test_message_with_feedback_negative(self):
        """Test message with negative feedback."""
        msg = Message(role="assistant", content="Answer", feedback="negative")
        
        assert msg.feedback == "negative"
    
    def test_message_with_branch_id(self):
        """Test message with branch ID."""
        msg = Message(role="user", content="Test", branch_id="branch-1")
        
        assert msg.branch_id == "branch-1"
    
    def test_message_with_parent_message_id(self):
        """Test message with parent message ID."""
        msg = Message(role="user", content="Test", parent_message_id="parent-1")
        
        assert msg.parent_message_id == "parent-1"


@pytest.mark.unit
class TestBranchDataclass:
    """Test Branch dataclass."""
    
    def test_branch_creation(self):
        """Test creating a branch."""
        branch = Branch(
            id="branch-1",
            name="Test Branch",
            created_at=datetime.now(),
            created_from_message_id="msg-1",
            message_count=5,
            is_active=True
        )
        
        assert branch.id == "branch-1"
        assert branch.name == "Test Branch"
        assert branch.message_count == 5
        assert branch.is_active is True
    
    def test_branch_default_active(self):
        """Test branch default is_active."""
        branch = Branch(
            id="branch-1",
            name="Test",
            created_at=datetime.now(),
            created_from_message_id="msg-1",
            message_count=1
        )
        
        assert branch.is_active is True
