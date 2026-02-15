"""Integration tests for complete workflows."""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


@pytest.mark.integration
class TestDocumentProcessingPipeline:
    """Test complete document processing pipeline."""
    
    @patch('rag.embeddings.SentenceTransformer')
    @patch('rag.vector_store.faiss')
    def test_full_document_pipeline(self, mock_faiss, mock_transformer_class):
        """Test complete document processing pipeline."""
        from rag.engine import RAGEngine
        
        # Setup mocks
        mock_model = Mock()
        mock_model.encode.return_value = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer_class.return_value = mock_model
        
        mock_index = Mock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.normalize_L2 = Mock()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is test content for the document pipeline.")
            temp_file = f.name
        
        try:
            engine = RAGEngine()
            success, message = engine.add_document(temp_file)
            
            assert success is True
            assert len(engine.documents) > 0
        finally:
            os.unlink(temp_file)
    
    def test_chunk_to_embedding_pipeline(self):
        """Test chunk to embedding pipeline."""
        from rag.chunker import chunk_text
        
        # Mock the embedding service to avoid loading real model
        with patch('rag.embeddings.SentenceTransformer') as mock_transformer:
            from rag.embeddings import EmbeddingService
            
            mock_model = Mock()
            mock_embeddings = Mock()
            mock_model.encode.return_value = mock_embeddings
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_transformer.return_value = mock_model
            
            # Create chunks
            text = "This is paragraph one.\n\nThis is paragraph two.\n\nThis is paragraph three."
            chunks = chunk_text(text, document_id="doc-1", document_name="test.txt", chunk_size=50)
            
            assert len(chunks) > 0
            
            # Generate embeddings
            service = EmbeddingService()
            texts = [chunk["text"] for chunk in chunks]
            embeddings = service.encode(texts)
            
            assert embeddings is not None


@pytest.mark.integration
class TestQueryRetrievalGeneration:
    """Test query to response pipeline."""
    
    @patch('core.workflow.validate_user_input')
    @patch('core.workflow.get_session_manager')
    @patch('core.workflow.RAGEngine')
    @patch('core.workflow.ModelHandler')
    def test_query_to_response_pipeline(
        self, mock_model_class, mock_rag_class, mock_get_session, mock_validate
    ):
        """Test complete query to response pipeline."""
        from core.workflow import WorkflowEngine
        
        mock_validate.return_value = (True, "")
        
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        
        mock_rag = Mock()
        mock_rag.get_context_string.return_value = "Retrieved context from documents"
        mock_rag.retrieve.return_value = [
            {"document_name": "doc1.txt", "text": "Relevant content", "score": 0.95},
        ]
        mock_rag_class.return_value = mock_rag
        
        mock_model = Mock()
        mock_model._prompt_builder.chat_prompt.return_value = "Formatted prompt with context"
        mock_model.generate_stream.return_value = iter([
            "Based", " ", "on", " ", "the", " ", "context", ",", " ", "this", " ", "is", " ", "the", " ", "answer", "."
        ])
        mock_model.extract_reasoning.return_value = (
            "I analyzed the context and found relevant information.",
            "Based on the context, this is the answer."
        )
        mock_model_class.return_value = mock_model
        
        engine = WorkflowEngine()
        
        # Collect all results
        results = list(engine.process_query("What is the answer?"))
        
        # Should have: status, retrieval, status, tokens, complete
        types = [r["type"] for r in results]
        assert "status" in types
        assert "retrieval" in types
        assert "token" in types
        assert "complete" in types
        
        # Verify complete result
        complete_results = [r for r in results if r["type"] == "complete"]
        assert len(complete_results) == 1
        assert "response" in complete_results[0]
        assert "reasoning" in complete_results[0]


@pytest.mark.integration
class TestSessionWorkflowIntegration:
    """Test session and workflow integration."""
    
    def test_conversation_history_in_prompt(self):
        """Test that conversation history is included in prompts."""
        from core.session import SessionManager
        from model.prompts import build_rag_prompt
        
        session = SessionManager()
        
        # Add conversation history
        session.add_user_message("What is AI?")
        session.add_assistant_message("AI is artificial intelligence.")
        session.add_user_message("What are its applications?")
        
        # Get history
        history = session.get_conversation_history()
        
        # Build prompt with history
        prompt = build_rag_prompt("Tell me more", "Context here", history)
        
        assert "What is AI?" in prompt
        assert "artificial intelligence" in prompt
        assert "applications" in prompt
    
    def test_session_persistence_across_operations(self):
        """Test session persists across workflow operations."""
        from core.session import SessionManager
        
        session = SessionManager()
        
        # Perform multiple operations
        session.add_user_message("Q1")
        session.add_assistant_message("A1")
        session.add_user_message("Q2")
        session.add_assistant_message("A2")
        session.register_document("doc-1")
        
        # Verify state
        assert session.message_count == 4
        assert "doc-1" in session._document_ids
        
        # Export and verify
        exported = session.export_conversation()
        assert len(exported["messages"]) == 4
        assert "doc-1" in exported["document_ids"]


@pytest.mark.integration
class TestRAGVectorStoreIntegration:
    """Test RAG and vector store integration."""
    
    @patch('rag.embeddings.SentenceTransformer')
    @patch('rag.vector_store.faiss')
    def test_add_and_retrieve_documents(self, mock_faiss, mock_transformer_class):
        """Test adding and retrieving documents."""
        from rag.engine import RAGEngine
        import numpy as np
        
        # Setup mocks
        mock_model = Mock()
        # Return normalized embeddings for similarity
        mock_model.encode.return_value = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        mock_model.encode_query.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_transformer_class.return_value = mock_model
        
        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[1.0, 0.5]]),  # scores
            np.array([[0, 1]])       # indices
        )
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.normalize_L2 = Mock()
        
        engine = RAGEngine()
        
        # Add document (mocked)
        mock_doc = Mock()
        mock_doc.id = "doc-1"
        mock_doc.name = "test.txt"
        mock_doc.content = "Test content"
        mock_doc.size_bytes = 100
        mock_doc.chunks = [
            {"id": "c1", "text": "Chunk 1", "document_id": "doc-1", "document_name": "test.txt"},
            {"id": "c2", "text": "Chunk 2", "document_id": "doc-1", "document_name": "test.txt"},
        ]
        engine.documents["doc-1"] = mock_doc
        
        # Manually add to vector store metadata
        engine.vector_store._metadata = {
            0: {"chunk_id": "c1", "document_id": "doc-1", "document_name": "test.txt", "text": "Chunk 1"},
            1: {"chunk_id": "c2", "document_id": "doc-1", "document_name": "test.txt", "text": "Chunk 2"},
        }
        engine.vector_store._id_counter = 2
        
        # Retrieve
        results = engine.retrieve("test query")
        
        assert len(results) > 0


@pytest.mark.functional
class TestEndToEndFunctional:
    """Functional end-to-end tests."""
    
    def test_document_upload_to_chat_response(self, temp_dir):
        """Test complete flow from upload to chat response."""
        from utils.file_utils import save_uploaded_file
        from utils.validators import validate_file_extension, validate_file_size
        
        # Create mock uploaded file
        mock_file = Mock()
        mock_file.name = "test.txt"
        mock_file.getbuffer.return_value = b"This is a test document about AI and machine learning."
        
        # Validate file
        assert validate_file_extension(mock_file.name) is True
        assert validate_file_size(len(mock_file.getbuffer())) is True
        
        # Save file
        success, file_path = save_uploaded_file(mock_file, temp_dir)
        assert success is True
        assert os.path.exists(file_path)
    
    def test_multi_turn_conversation(self):
        """Test multi-turn conversation flow."""
        from core.session import SessionManager
        from model.prompts import build_rag_prompt
        
        session = SessionManager()
        
        # Simulate multi-turn conversation
        conversation = [
            ("user", "What is machine learning?"),
            ("assistant", "Machine learning is a subset of AI..."),
            ("user", "What are neural networks?"),
            ("assistant", "Neural networks are computational models..."),
            ("user", "How do they relate to deep learning?"),
        ]
        
        for role, content in conversation:
            if role == "user":
                session.add_user_message(content)
            else:
                session.add_assistant_message(content)
        
        # Get history
        history = session.get_conversation_history(max_turns=3)
        
        # Build prompt
        prompt = build_rag_prompt("Latest question", "Context", history)
        
        # Verify conversation continuity
        assert "machine learning" in prompt.lower()
        assert "neural networks" in prompt.lower()
        assert len(history) == 5  # 5 messages total


@pytest.mark.integration
class TestErrorRecoveryIntegration:
    """Test error recovery in integrated workflows."""
    
    @patch('core.workflow.validate_user_input')
    @patch('core.workflow.get_session_manager')
    @patch('core.workflow.RAGEngine')
    @patch('core.workflow.ModelHandler')
    def test_retrieval_failure_recovery(
        self, mock_model_class, mock_rag_class, mock_get_session, mock_validate
    ):
        """Test recovery from retrieval failure."""
        from core.workflow import WorkflowEngine
        
        mock_validate.return_value = (True, "")
        
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        
        mock_rag = Mock()
        # Simulate retrieval failure
        mock_rag.get_context_string.side_effect = Exception("Vector store error")
        mock_rag.retrieve.side_effect = Exception("Search failed")
        mock_rag_class.return_value = mock_rag
        
        mock_model = Mock()
        mock_model.generate_stream.return_value = iter(["Fallback", " ", "response"])
        mock_model.extract_reasoning.return_value = ("", "Fallback response")
        mock_model_class.return_value = mock_model
        
        engine = WorkflowEngine()
        
        # Should handle error gracefully
        results = list(engine.process_query("test query"))
        
        # Should still complete with fallback
        complete_results = [r for r in results if r["type"] == "complete"]
        assert len(complete_results) > 0
    
    def test_partial_document_processing_recovery(self):
        """Test recovery from partial document processing failure."""
        from rag.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # Test with corrupted/malformed content
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as f:
            # Write some binary content that might cause issues
            f.write(b"\x00\x01\x02\x03Regular text")
            temp_file = f.name
        
        try:
            # Should handle gracefully
            doc = processor.process(temp_file)
            assert doc is not None
        finally:
            os.unlink(temp_file)
