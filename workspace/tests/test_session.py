"""Unit tests for core.session module."""

import pytest
from datetime import datetime
from uuid import UUID

from core.session import Message, SessionManager, get_session_manager


@pytest.mark.unit
class TestMessage:
    """Test Message dataclass."""
    
    def test_message_creation(self):
        """Test Message creation."""
        msg = Message(role="user", content="Hello")
        
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert isinstance(msg.timestamp, datetime)
        assert msg.reasoning is None
        assert msg.sources is None
        assert isinstance(UUID(msg.id), UUID)
    
    def test_message_with_reasoning(self):
        """Test Message with reasoning."""
        msg = Message(
            role="assistant",
            content="Answer",
            reasoning="Step 1: Think\nStep 2: Answer",
            sources=["doc1.txt", "doc2.txt"]
        )
        
        assert msg.reasoning == "Step 1: Think\nStep 2: Answer"
        assert msg.sources == ["doc1.txt", "doc2.txt"]


@pytest.mark.unit
class TestSessionManagerInitialization:
    """Test SessionManager initialization."""
    
    def test_initialization(self):
        """Test SessionManager initialization."""
        manager = SessionManager()
        
        assert manager._messages == []
        assert isinstance(UUID(manager._session_id), UUID)
        assert isinstance(manager._created_at, datetime)
        assert manager._document_ids == set()
    
    def test_message_count(self):
        """Test message_count property."""
        manager = SessionManager()
        
        assert manager.message_count == 0
        
        manager.add_user_message("Hello")
        assert manager.message_count == 1
        
        manager.add_assistant_message("Hi")
        assert manager.message_count == 2
    
    def test_is_empty(self):
        """Test is_empty property."""
        manager = SessionManager()
        
        assert manager.is_empty is True
        
        manager.add_user_message("Hello")
        assert manager.is_empty is False


@pytest.mark.unit
class TestSessionManagerAddMessages:
    """Test adding messages."""
    
    def test_add_user_message(self):
        """Test adding user message."""
        manager = SessionManager()
        msg = manager.add_user_message("Hello")
        
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert len(manager._messages) == 1
    
    def test_add_assistant_message(self):
        """Test adding assistant message."""
        manager = SessionManager()
        msg = manager.add_assistant_message("Hi there!")
        
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"
    
    def test_add_assistant_message_with_reasoning(self):
        """Test adding assistant message with reasoning."""
        manager = SessionManager()
        msg = manager.add_assistant_message(
            content="The answer is 42.",
            reasoning="Based on analysis...",
            sources=["doc1.txt"]
        )
        
        assert msg.reasoning == "Based on analysis..."
        assert msg.sources == ["doc1.txt"]
    
    def test_add_system_message(self):
        """Test adding system message."""
        manager = SessionManager()
        msg = manager.add_system_message("System initialization")
        
        assert msg.role == "system"
        assert msg.content == "System initialization"


@pytest.mark.unit
class TestSessionManagerGetMessages:
    """Test getting messages."""
    
    def test_get_all_messages(self):
        """Test getting all messages."""
        manager = SessionManager()
        manager.add_user_message("Q1")
        manager.add_assistant_message("A1")
        manager.add_user_message("Q2")
        
        messages = manager.get_messages()
        
        assert len(messages) == 3
        assert messages[0].content == "Q1"
        assert messages[2].content == "Q2"
    
    def test_get_messages_with_max_turns(self):
        """Test getting limited messages."""
        manager = SessionManager()
        
        # Add 5 turns (10 messages)
        for i in range(5):
            manager.add_user_message(f"Q{i}")
            manager.add_assistant_message(f"A{i}")
        
        messages = manager.get_messages(max_turns=3)
        
        # Should get last 3 turns = 6 messages
        assert len(messages) == 6
        assert messages[0].content == "Q2"
    
    def test_get_conversation_history(self):
        """Test getting conversation history as dicts."""
        manager = SessionManager()
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi", reasoning="Thinking...")
        
        history = manager.get_conversation_history()
        
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert "timestamp" in history[0]
        assert history[1]["reasoning"] == "Thinking..."


@pytest.mark.unit
class TestSessionManagerFormattedHistory:
    """Test formatted history."""
    
    def test_formatted_history_for_model(self):
        """Test history formatted for model."""
        manager = SessionManager()
        manager.add_user_message("What is AI?")
        manager.add_assistant_message("AI is artificial intelligence.")
        
        formatted = manager.get_formatted_history(for_model=True)
        
        assert "User: What is AI?" in formatted
        assert "Assistant: AI is artificial intelligence." in formatted
    
    def test_formatted_history_for_display(self):
        """Test history formatted for display."""
        manager = SessionManager()
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")
        
        formatted = manager.get_formatted_history(for_model=False)
        
        assert "🧑" in formatted
        assert "🤖" in formatted
        assert "Hello" in formatted
        assert "Hi" in formatted


@pytest.mark.unit
class TestSessionManagerClear:
    """Test clearing conversation."""
    
    def test_clear_conversation(self):
        """Test clearing conversation."""
        manager = SessionManager()
        manager.add_user_message("Q1")
        manager.add_assistant_message("A1")
        
        manager.clear_conversation()
        
        assert len(manager._messages) == 0
        assert manager.is_empty is True


@pytest.mark.unit
class TestSessionManagerUpdateMessage:
    """Test updating last message."""
    
    def test_update_last_assistant_message(self):
        """Test updating last assistant message."""
        manager = SessionManager()
        manager.add_user_message("Question")
        manager.add_assistant_message("Initial answer")
        
        result = manager.update_last_message(
            content="Updated answer",
            reasoning="Updated reasoning"
        )
        
        assert result is True
        assert manager.get_last_message().content == "Updated answer"
        assert manager.get_last_message().reasoning == "Updated reasoning"
    
    def test_update_last_user_message_fails(self):
        """Test updating last message when it's user message fails."""
        manager = SessionManager()
        manager.add_user_message("Question")
        
        result = manager.update_last_message(content="Updated")
        
        assert result is False
    
    def test_update_empty_conversation(self):
        """Test updating empty conversation."""
        manager = SessionManager()
        
        result = manager.update_last_message(content="Updated")
        
        assert result is False


@pytest.mark.unit
class TestSessionManagerDocuments:
    """Test document registration."""
    
    def test_register_document(self):
        """Test registering a document."""
        manager = SessionManager()
        manager.register_document("doc-1")
        
        assert "doc-1" in manager._document_ids
    
    def test_unregister_document(self):
        """Test unregistering a document."""
        manager = SessionManager()
        manager.register_document("doc-1")
        manager.unregister_document("doc-1")
        
        assert "doc-1" not in manager._document_ids
    
    def test_get_session_info(self):
        """Test getting session info."""
        manager = SessionManager()
        manager.register_document("doc-1")
        manager.add_user_message("Hello")
        
        info = manager.get_session_info()
        
        assert "session_id" in info
        assert "created_at" in info
        assert info["message_count"] == 1
        assert info["document_count"] == 1
        assert "duration_minutes" in info


@pytest.mark.unit
class TestSessionManagerExportImport:
    """Test export and import functionality."""
    
    def test_export_conversation(self):
        """Test exporting conversation."""
        manager = SessionManager()
        manager.add_user_message("Q1")
        manager.add_assistant_message("A1", reasoning="R1")
        manager.register_document("doc-1")
        
        exported = manager.export_conversation()
        
        assert "session_id" in exported
        assert "created_at" in exported
        assert "messages" in exported
        assert len(exported["messages"]) == 2
        assert exported["document_ids"] == ["doc-1"]
    
    def test_import_conversation(self):
        """Test importing conversation."""
        data = {
            "session_id": "test-session",
            "created_at": datetime.now().isoformat(),
            "messages": [
                {
                    "id": "msg-1",
                    "role": "user",
                    "content": "Hello",
                    "timestamp": datetime.now().isoformat(),
                    "reasoning": None,
                    "sources": None,
                },
                {
                    "id": "msg-2",
                    "role": "assistant",
                    "content": "Hi",
                    "timestamp": datetime.now().isoformat(),
                    "reasoning": "Thinking...",
                    "sources": ["doc.txt"],
                }
            ],
            "document_ids": ["doc-1"],
        }
        
        manager = SessionManager()
        result = manager.import_conversation(data)
        
        assert result is True
        assert manager.message_count == 2
        assert "doc-1" in manager._document_ids
    
    def test_import_invalid_data(self):
        """Test importing invalid data."""
        manager = SessionManager()
        result = manager.import_conversation({"invalid": "data"})
        
        assert result is True  # Currently doesn't validate strictly


@pytest.mark.unit
class TestGetSessionManager:
    """Test get_session_manager singleton."""
    
    def test_singleton(self):
        """Test that get_session_manager returns singleton."""
        # Reset singleton
        import core.session as sess_module
        sess_module._session_manager = None
        
        manager1 = get_session_manager()
        manager2 = get_session_manager()
        
        assert manager1 is manager2
