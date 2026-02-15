"""Unit tests for core.export module."""

import pytest
import json
from datetime import datetime

from core.export import ExportService, get_export_service
from core.session import Message


@pytest.mark.unit
class TestExportServiceInitialization:
    """Test ExportService initialization."""
    
    def test_initialization(self):
        """Test ExportService initialization."""
        service = ExportService()
        
        assert service is not None


@pytest.mark.unit
class TestExportServiceMarkdown:
    """Test Markdown export functionality."""
    
    def test_export_markdown_empty(self):
        """Test exporting empty messages."""
        service = ExportService()
        
        result = service.export_markdown([])
        
        assert "# Chat History" in result
        assert "Exported:" in result
    
    def test_export_markdown_single_message(self):
        """Test exporting single message."""
        service = ExportService()
        
        messages = [
            Message(role="user", content="Hello")
        ]
        
        result = service.export_markdown(messages)
        
        assert "# Chat History" in result
        assert "user" in result.lower()
        assert "Hello" in result
    
    def test_export_markdown_multiple_messages(self):
        """Test exporting multiple messages."""
        service = ExportService()
        
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there")
        ]
        
        result = service.export_markdown(messages)
        
        assert "user" in result.lower()
        assert "assistant" in result.lower()
        assert "Hello" in result
        assert "Hi there" in result
    
    def test_export_markdown_without_metadata(self):
        """Test exporting without metadata."""
        service = ExportService()
        
        messages = [
            Message(role="user", content="Hello")
        ]
        
        result = service.export_markdown(messages, include_metadata=False)
        
        assert "Timestamp:" not in result
    
    def test_export_markdown_with_metadata(self):
        """Test exporting with metadata."""
        service = ExportService()
        
        msg = Message(role="user", content="Hello")
        messages = [msg]
        
        result = service.export_markdown(messages, include_metadata=True)
        
        assert "Timestamp:" in result
        assert "ID:" in result
    
    def test_export_markdown_pinned_message(self):
        """Test exporting pinned message."""
        service = ExportService()
        
        msg = Message(role="user", content="Important", is_pinned=True)
        messages = [msg]
        
        result = service.export_markdown(messages)
        
        assert "📌" in result or "Pinned" in result
    
    def test_export_markdown_with_feedback(self):
        """Test exporting message with feedback."""
        service = ExportService()
        
        msg = Message(role="assistant", content="Answer", feedback="positive")
        messages = [msg]
        
        result = service.export_markdown(messages)
        
        assert "👍" in result or "Feedback" in result
    
    def test_export_markdown_with_reasoning(self):
        """Test exporting message with reasoning."""
        service = ExportService()
        
        msg = Message(
            role="assistant",
            content="Answer",
            reasoning="Step by step"
        )
        messages = [msg]
        
        result = service.export_markdown(messages, include_metadata=True)
        
        assert "Reasoning:" in result
    
    def test_export_markdown_with_sources(self):
        """Test exporting message with sources."""
        service = ExportService()
        
        msg = Message(
            role="assistant",
            content="Answer",
            sources=["doc1.txt", "doc2.txt"]
        )
        messages = [msg]
        
        result = service.export_markdown(messages, include_metadata=True)
        
        assert "Sources:" in result
        assert "doc1.txt" in result


@pytest.mark.unit
class TestExportServiceJSON:
    """Test JSON export functionality."""
    
    def test_export_json_empty(self):
        """Test exporting empty messages to JSON."""
        service = ExportService()
        
        result = service.export_json([])
        
        data = json.loads(result)
        
        assert "exported_at" in data
        assert data["message_count"] == 0
        assert data["messages"] == []
    
    def test_export_json_single_message(self):
        """Test exporting single message to JSON."""
        service = ExportService()
        
        messages = [
            Message(role="user", content="Hello")
        ]
        
        result = service.export_json(messages)
        
        data = json.loads(result)
        
        assert data["message_count"] == 1
        assert data["messages"][0]["content"] == "Hello"
        assert data["messages"][0]["role"] == "user"
    
    def test_export_json_without_metadata(self):
        """Test exporting JSON without metadata."""
        service = ExportService()
        
        messages = [
            Message(
                role="assistant",
                content="Answer",
                reasoning="Think",
                sources=["doc.txt"],
                is_pinned=True,
                feedback="positive"
            )
        ]
        
        result = service.export_json(messages, include_metadata=False)
        
        data = json.loads(result)
        
        assert "reasoning" not in data["messages"][0]
        assert "sources" not in data["messages"][0]
    
    def test_export_json_with_metadata(self):
        """Test exporting JSON with metadata."""
        service = ExportService()
        
        msg = Message(
            role="assistant",
            content="Answer",
            reasoning="Think",
            sources=["doc.txt"],
            is_pinned=True,
            feedback="positive",
            branch_id="branch-1",
            parent_message_id="parent-1"
        )
        messages = [msg]
        
        result = service.export_json(messages, include_metadata=True)
        
        data = json.loads(result)
        
        msg_data = data["messages"][0]
        assert msg_data["reasoning"] == "Think"
        assert "doc.txt" in msg_data["sources"]
        assert msg_data["is_pinned"] == "True"
        assert msg_data["feedback"] == "positive"
        assert msg_data["branch_id"] == "branch-1"
        assert msg_data["parent_message_id"] == "parent-1"


@pytest.mark.unit
class TestExportServicePlainText:
    """Test plain text export functionality."""
    
    def test_export_plain_text_empty(self):
        """Test exporting empty messages to plain text."""
        service = ExportService()
        
        result = service.export_plain_text([])
        
        assert result == ""
    
    def test_export_plain_text_single_message(self):
        """Test exporting single message to plain text."""
        service = ExportService()
        
        messages = [
            Message(role="user", content="Hello")
        ]
        
        result = service.export_plain_text(messages)
        
        assert "USER:" in result
        assert "Hello" in result
    
    def test_export_plain_text_multiple_messages(self):
        """Test exporting multiple messages to plain text."""
        service = ExportService()
        
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi")
        ]
        
        result = service.export_plain_text(messages)
        
        assert "USER:" in result
        assert "ASSISTANT:" in result
    
    def test_export_plain_text_timestamps(self):
        """Test plain text includes timestamps."""
        service = ExportService()
        
        msg = Message(role="user", content="Hello")
        messages = [msg]
        
        result = service.export_plain_text(messages)
        
        assert msg.timestamp.strftime('%Y-%m-%d') in result


@pytest.mark.unit
class TestExportServiceTimestamps:
    """Test export with timestamps functionality."""
    
    def test_export_with_timestamps_markdown(self):
        """Test exporting with timestamps as markdown."""
        service = ExportService()
        
        messages = [
            Message(role="user", content="Hello")
        ]
        
        result = service.export_with_timestamps(messages, format="markdown")
        
        assert "# Chat History" in result
        assert "Timestamp:" in result
    
    def test_export_with_timestamps_json(self):
        """Test exporting with timestamps as JSON."""
        service = ExportService()
        
        messages = [
            Message(role="user", content="Hello")
        ]
        
        result = service.export_with_timestamps(messages, format="json")
        
        data = json.loads(result)
        
        assert "exported_at" in data
    
    def test_export_with_timestamps_plain(self):
        """Test exporting with timestamps as plain text."""
        service = ExportService()
        
        messages = [
            Message(role="user", content="Hello")
        ]
        
        result = service.export_with_timestamps(messages, format="plain")
        
        assert "USER:" in result


@pytest.mark.unit
class TestExportServiceBranchInfo:
    """Test export with branch information."""
    
    def test_export_with_branch_info(self):
        """Test exporting with branch information."""
        from core.session import Branch
        
        service = ExportService()
        
        messages = [
            Message(
                role="user",
                content="Hello",
                branch_id="branch-1",
                parent_message_id="parent-1"
            )
        ]
        
        branches = [
            Branch(
                id="branch-1",
                name="Branch 1",
                created_at=datetime.now(),
                created_from_message_id="parent-1",
                message_count=1,
                is_active=True
            )
        ]
        
        result = service.export_with_branch_info(messages, branches)
        
        data = json.loads(result)
        
        assert "branches" in data
        assert len(data["branches"]) == 1
        assert data["branches"][0]["name"] == "Branch 1"


@pytest.mark.unit
class TestGetExportService:
    """Test get_export_service singleton."""
    
    def test_singleton(self):
        """Test get_export_service returns singleton."""
        import core.export as export_module
        export_module._export_service = None
        
        service1 = get_export_service()
        service2 = get_export_service()
        
        assert service1 is service2
