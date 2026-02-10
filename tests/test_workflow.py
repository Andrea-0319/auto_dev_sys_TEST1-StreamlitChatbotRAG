"""Unit tests for core.workflow module."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from core.workflow import WorkflowEngine, create_workflow


@pytest.mark.unit
class TestWorkflowEngineInitialization:
    """Test WorkflowEngine initialization."""
    
    @patch('core.workflow.get_session_manager')
    @patch('core.workflow.RAGEngine')
    @patch('core.workflow.ModelHandler')
    def test_initialization_defaults(self, mock_model_class, mock_rag_class, mock_get_session):
        """Test WorkflowEngine initialization with defaults."""
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        
        mock_rag = Mock()
        mock_rag_class.return_value = mock_rag
        
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        engine = WorkflowEngine()
        
        assert engine.session is mock_session
        assert engine.rag is mock_rag
        assert engine.model is mock_model
    
    @patch('core.workflow.get_session_manager')
    def test_initialization_with_instances(self, mock_get_session):
        """Test WorkflowEngine with provided instances."""
        mock_session = Mock()
        mock_rag = Mock()
        mock_model = Mock()
        
        engine = WorkflowEngine(
            session_manager=mock_session,
            rag_engine=mock_rag,
            model_handler=mock_model
        )
        
        assert engine.session is mock_session
        assert engine.rag is mock_rag
        assert engine.model is mock_model


@pytest.mark.unit
class TestWorkflowEngineProcessQuery:
    """Test query processing workflow."""
    
    @patch('core.workflow.validate_user_input')
    @patch('core.workflow.get_session_manager')
    def test_process_query_invalid_input(self, mock_get_session, mock_validate):
        """Test processing invalid query."""
        mock_validate.return_value = (False, "Input too long")
        mock_get_session.return_value = Mock()
        
        engine = WorkflowEngine()
        results = list(engine.process_query("test"))
        
        assert len(results) == 1
        assert results[0]["type"] == "error"
        assert "too long" in results[0]["message"]
    
    @patch('core.workflow.validate_user_input')
    @patch('core.workflow.get_session_manager')
    @patch('core.workflow.RAGEngine')
    @patch('core.workflow.ModelHandler')
    def test_process_query_with_retrieval(
        self, mock_model_class, mock_rag_class, mock_get_session, mock_validate
    ):
        """Test query processing with retrieval."""
        mock_validate.return_value = (True, "")
        
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        
        mock_rag = Mock()
        mock_rag.get_context_string.return_value = "Retrieved context"
        mock_rag.retrieve.return_value = [
            {"document_name": "doc1.txt", "text": "Content 1", "score": 0.9},
        ]
        mock_rag_class.return_value = mock_rag
        
        mock_model = Mock()
        mock_model._prompt_builder.chat_prompt.return_value = "Formatted prompt"
        mock_model.generate_stream.return_value = iter(["Hello", " ", "World"])
        mock_model.extract_reasoning.return_value = ("Reasoning", "Hello World")
        mock_model_class.return_value = mock_model
        
        engine = WorkflowEngine()
        results = list(engine.process_query("test query"))
        
        # Should yield status, retrieval, status, tokens, complete
        assert any(r["type"] == "status" for r in results)
        assert any(r["type"] == "retrieval" for r in results)
        assert any(r["type"] == "complete" for r in results)
    
    @patch('core.workflow.validate_user_input')
    @patch('core.workflow.get_session_manager')
    @patch('core.workflow.RAGEngine')
    @patch('core.workflow.ModelHandler')
    def test_process_query_generation_error(
        self, mock_model_class, mock_rag_class, mock_get_session, mock_validate
    ):
        """Test handling generation error."""
        mock_validate.return_value = (True, "")
        
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        
        mock_rag = Mock()
        mock_rag.get_context_string.return_value = ""
        mock_rag.retrieve.return_value = []
        mock_rag_class.return_value = mock_rag
        
        mock_model = Mock()
        mock_model._prompt_builder.chat_prompt.return_value = "prompt"
        mock_model.generate_stream.side_effect = Exception("Generation failed")
        mock_model_class.return_value = mock_model
        
        engine = WorkflowEngine()
        results = list(engine.process_query("test"))
        
        error_results = [r for r in results if r["type"] == "error"]
        assert len(error_results) > 0
        assert "failed" in error_results[0]["message"].lower()


@pytest.mark.unit
class TestWorkflowEngineDocumentManagement:
    """Test document management in workflow."""
    
    @patch('core.workflow.get_session_manager')
    @patch('core.workflow.RAGEngine')
    def test_upload_document(self, mock_rag_class, mock_get_session):
        """Test document upload."""
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        
        mock_rag = Mock()
        mock_rag.add_document.return_value = (True, "Document added successfully")
        mock_rag_class.return_value = mock_rag
        
        engine = WorkflowEngine()
        success, message = engine.upload_document("test.txt")
        
        assert success is True
        assert "added" in message.lower()
    
    @patch('core.workflow.get_session_manager')
    @patch('core.workflow.RAGEngine')
    def test_upload_document_failure(self, mock_rag_class, mock_get_session):
        """Test document upload failure."""
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        
        mock_rag = Mock()
        mock_rag.add_document.return_value = (False, "Invalid file type")
        mock_rag_class.return_value = mock_rag
        
        engine = WorkflowEngine()
        success, message = engine.upload_document("test.jpg")
        
        assert success is False
        assert "invalid" in message.lower() or "failed" in message.lower()
    
    @patch('core.workflow.get_session_manager')
    @patch('core.workflow.RAGEngine')
    def test_remove_document(self, mock_rag_class, mock_get_session):
        """Test document removal."""
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        
        mock_rag = Mock()
        mock_rag.remove_document.return_value = True
        mock_rag_class.return_value = mock_rag
        
        engine = WorkflowEngine()
        result = engine.remove_document("doc-1")
        
        assert result is True
    
    @patch('core.workflow.get_session_manager')
    @patch('core.workflow.RAGEngine')
    def test_get_document_list(self, mock_rag_class, mock_get_session):
        """Test getting document list."""
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        
        mock_rag = Mock()
        mock_rag.get_document_list.return_value = [
            {"id": "doc-1", "name": "file1.txt"},
            {"id": "doc-2", "name": "file2.txt"},
        ]
        mock_rag_class.return_value = mock_rag
        
        engine = WorkflowEngine()
        docs = engine.get_document_list()
        
        assert len(docs) == 2
        assert docs[0]["name"] == "file1.txt"


@pytest.mark.unit
class TestWorkflowEngineSessionManagement:
    """Test session management in workflow."""
    
    @patch('core.workflow.get_session_manager')
    @patch('core.workflow.RAGEngine')
    def test_clear_conversation(self, mock_rag_class, mock_get_session):
        """Test clearing conversation."""
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        
        mock_rag = Mock()
        mock_rag_class.return_value = mock_rag
        
        engine = WorkflowEngine()
        engine.clear_conversation()
        
        mock_session.clear_conversation.assert_called_once()
    
    @patch('core.workflow.get_session_manager')
    @patch('core.workflow.RAGEngine')
    def test_clear_all_documents(self, mock_rag_class, mock_get_session):
        """Test clearing all documents."""
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        
        mock_rag = Mock()
        mock_rag.clear_all.return_value = True
        mock_rag_class.return_value = mock_rag
        
        engine = WorkflowEngine()
        result = engine.clear_all_documents()
        
        assert result is True


@pytest.mark.unit
class TestWorkflowEngineStats:
    """Test workflow statistics."""
    
    @patch('core.workflow.get_session_manager')
    @patch('core.workflow.RAGEngine')
    @patch('core.workflow.ModelHandler')
    def test_get_stats(self, mock_model_class, mock_rag_class, mock_get_session):
        """Test getting workflow stats."""
        mock_session = Mock()
        mock_session.get_session_info.return_value = {"message_count": 5}
        mock_get_session.return_value = mock_session
        
        mock_rag = Mock()
        mock_rag.get_stats.return_value = {"vector_count": 10}
        mock_rag_class.return_value = mock_rag
        
        mock_model = Mock()
        mock_model.get_model_info.return_value = {"loaded": True}
        mock_model_class.return_value = mock_model
        
        engine = WorkflowEngine()
        stats = engine.get_stats()
        
        assert "session" in stats
        assert "rag" in stats
        assert "model" in stats
        assert stats["session"]["message_count"] == 5


@pytest.mark.unit
class TestCreateWorkflow:
    """Test create_workflow convenience function."""
    
    def test_create_workflow(self):
        """Test create_workflow function."""
        workflow = create_workflow()
        
        assert isinstance(workflow, WorkflowEngine)
