"""Unit tests for core.branch_manager module."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from core.branch_manager import BranchManager, get_branch_manager
from core.session import Message, Branch, SessionManager


@pytest.mark.unit
class TestBranchManagerInitialization:
    """Test BranchManager initialization."""
    
    def test_initialization(self):
        """Test BranchManager initialization."""
        session_manager = SessionManager()
        
        manager = BranchManager(session_manager)
        
        assert manager._session_manager is session_manager
        assert manager._branch_data == {}


@pytest.mark.unit
class TestBranchManagerCreateBranch:
    """Test branch creation functionality."""
    
    def test_create_branch_basic(self):
        """Test creating a basic branch."""
        session_manager = SessionManager()
        session_manager.add_user_message("Hello")
        session_manager.add_assistant_message("Hi there")
        
        manager = BranchManager(session_manager)
        
        messages = session_manager.get_messages()
        branch = manager.create_branch(messages[0])
        
        assert branch is not None
        assert branch.id is not None
        assert branch.name is not None
    
    def test_create_branch_with_name(self):
        """Test creating a branch with custom name."""
        session_manager = SessionManager()
        session_manager.add_user_message("Hello")
        
        manager = BranchManager(session_manager)
        
        messages = session_manager.get_messages()
        branch = manager.create_branch(messages[0], name="My Branch")
        
        assert branch.name == "My Branch"
    
    def test_create_branch_max_reached(self):
        """Test that max branches limit is enforced."""
        session_manager = SessionManager()
        
        for i in range(10):
            session_manager.add_user_message(f"Message {i}")
        
        manager = BranchManager(session_manager)
        
        messages = session_manager.get_messages()
        
        for i in range(10):
            result = manager.create_branch(messages[0], name=f"Branch {i}")
        
        extra_branch = manager.create_branch(messages[0], name="Extra Branch")
        
        assert extra_branch is None
    
    def test_create_branch_invalid_message(self):
        """Test creating branch from invalid message."""
        session_manager = SessionManager()
        session_manager.add_user_message("Hello")
        
        manager = BranchManager(session_manager)
        
        fake_message = Message(role="user", content="Fake")
        branch = manager.create_branch(fake_message)
        
        assert branch is None


@pytest.mark.unit
class TestBranchManagerListBranches:
    """Test listing branches."""
    
    def test_list_branches_empty(self):
        """Test listing branches when none exist."""
        session_manager = SessionManager()
        
        manager = BranchManager(session_manager)
        
        branches = manager.list_branches()
        
        assert branches == []
    
    def test_list_branches_multiple(self):
        """Test listing multiple branches."""
        session_manager = SessionManager()
        
        for i in range(5):
            session_manager.add_user_message(f"Message {i}")
        
        manager = BranchManager(session_manager)
        
        messages = session_manager.get_messages()
        
        for i in range(3):
            manager.create_branch(messages[0], name=f"Branch {i}")
        
        branches = manager.list_branches()
        
        assert len(branches) == 3


@pytest.mark.unit
class TestBranchManagerGetBranch:
    """Test getting specific branch."""
    
    def test_get_branch_exists(self):
        """Test getting an existing branch."""
        session_manager = SessionManager()
        session_manager.add_user_message("Hello")
        
        manager = BranchManager(session_manager)
        
        messages = session_manager.get_messages()
        created_branch = manager.create_branch(messages[0], name="Test Branch")
        
        retrieved = manager.get_branch(created_branch.id)
        
        assert retrieved is not None
        assert retrieved.id == created_branch.id
        assert retrieved.name == "Test Branch"
    
    def test_get_branch_not_exists(self):
        """Test getting non-existent branch."""
        session_manager = SessionManager()
        
        manager = BranchManager(session_manager)
        
        branch = manager.get_branch("non-existent-id")
        
        assert branch is None


@pytest.mark.unit
class TestBranchManagerDeleteBranch:
    """Test deleting branches."""
    
    def test_delete_branch(self):
        """Test deleting a branch."""
        session_manager = SessionManager()
        session_manager.add_user_message("Hello")
        
        manager = BranchManager(session_manager)
        
        messages = session_manager.get_messages()
        branch = manager.create_branch(messages[0])
        
        result = manager.delete_branch(branch.id)
        
        assert result is True
        
        deleted = manager.get_branch(branch.id)
        assert deleted is None
    
    def test_delete_branch_not_exists(self):
        """Test deleting non-existent branch."""
        session_manager = SessionManager()
        
        manager = BranchManager(session_manager)
        
        result = manager.delete_branch("non-existent-id")
        
        assert result is False


@pytest.mark.unit
class TestBranchManagerMergeBranch:
    """Test merging branches."""
    
    def test_merge_branch_success(self):
        """Test successfully merging branches."""
        session_manager = SessionManager()
        session_manager.add_user_message("Hello")
        session_manager.add_assistant_message("Hi")
        
        manager = BranchManager(session_manager)
        
        messages = session_manager.get_messages()
        
        branch1 = manager.create_branch(messages[0], name="Branch 1")
        branch2 = manager.create_branch(messages[0], name="Branch 2")
        
        result = manager.merge_branch(branch1.id, branch2.id)
        
        assert result is True
    
    def test_merge_branch_source_not_exists(self):
        """Test merging with non-existent source."""
        session_manager = SessionManager()
        session_manager.add_user_message("Hello")
        
        manager = BranchManager(session_manager)
        
        messages = session_manager.get_messages()
        branch = manager.create_branch(messages[0])
        
        result = manager.merge_branch("non-existent", branch.id)
        
        assert result is False
    
    def test_merge_branch_target_not_exists(self):
        """Test merging with non-existent target."""
        session_manager = SessionManager()
        session_manager.add_user_message("Hello")
        
        manager = BranchManager(session_manager)
        
        messages = session_manager.get_messages()
        branch = manager.create_branch(messages[0])
        
        result = manager.merge_branch(branch.id, "non-existent")
        
        assert result is False


@pytest.mark.unit
class TestBranchManagerGetBranchMessages:
    """Test getting messages for a branch."""
    
    def test_get_branch_messages(self):
        """Test getting messages for a branch."""
        session_manager = SessionManager()
        session_manager.add_user_message("Message 1")
        session_manager.add_assistant_message("Response 1")
        
        manager = BranchManager(session_manager)
        
        messages = session_manager.get_messages()
        branch = manager.create_branch(messages[0])
        
        branch_messages = manager.get_branch_messages(branch.id)
        
        assert len(branch_messages) > 0
    
    def test_get_branch_messages_empty(self):
        """Test getting messages for non-existent branch."""
        session_manager = SessionManager()
        
        manager = BranchManager(session_manager)
        
        messages = manager.get_branch_messages("non-existent")
        
        assert messages == []


@pytest.mark.unit
class TestBranchManagerGetBranchTree:
    """Test getting branch tree structure."""
    
    def test_get_branch_tree_empty(self):
        """Test getting tree with no branches."""
        session_manager = SessionManager()
        
        manager = BranchManager(session_manager)
        
        tree = manager.get_branch_tree()
        
        assert "branches" in tree
        assert "current_branch" in tree
        assert tree["branches"] == []
    
    def test_get_branch_tree_with_branches(self):
        """Test getting tree with branches."""
        session_manager = SessionManager()
        session_manager.add_user_message("Hello")
        
        manager = BranchManager(session_manager)
        
        messages = session_manager.get_messages()
        manager.create_branch(messages[0], name="Branch 1")
        manager.create_branch(messages[0], name="Branch 2")
        
        tree = manager.get_branch_tree()
        
        assert len(tree["branches"]) == 2
        assert tree["current_branch"] is not None


@pytest.mark.unit
class TestGetBranchManager:
    """Test get_branch_manager singleton."""
    
    def test_singleton(self):
        """Test get_branch_manager returns singleton."""
        import core.branch_manager as bm_module
        bm_module._branch_manager = None
        
        session_manager = SessionManager()
        
        manager1 = get_branch_manager(session_manager)
        manager2 = get_branch_manager(session_manager)
        
        assert manager1 is manager2
