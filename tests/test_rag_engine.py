"""Unit tests for rag.engine module."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from rag.engine import RAGEngine


@pytest.mark.unit
class TestRAGEngineInitialization:
    """Test RAGEngine initialization."""
    
    @patch('rag.engine.VectorStore')
    @patch('rag.engine.get_embedding_service')
    def test_initialization(self, mock_get_emb, mock_vector_store):
        """Test RAGEngine initialization."""
        mock_emb_service = Mock()
        mock_emb_service.dimension = 384
        mock_get_emb.return_value = mock_emb_service
        
        mock_store = Mock()
        mock_store.load.return_value = False
        mock_vector_store.return_value = mock_store
        
        engine = RAGEngine()
        
        assert engine.doc_processor is not None
        assert engine.embedding_service is not None
        assert engine.vector_store is not None
        assert engine.documents == {}


@pytest.mark.unit
class TestRAGEngineAddDocument:
    """Test adding documents to RAG engine."""
    
    @patch('rag.engine.VectorStore')
    @patch('rag.engine.get_embedding_service')
    @patch('rag.engine.validate_file_extension')
    @patch('rag.engine.validate_file_size')
    @patch('rag.engine.DocumentProcessor')
    @patch('rag.engine.chunk_text')
    def test_add_valid_document(
        self, mock_chunk, mock_processor_class, mock_validate_size,
        mock_validate_ext, mock_get_emb, mock_vector_store
    ):
        """Test adding a valid document."""
        # Setup mocks
        mock_validate_ext.return_value = True
        mock_validate_size.return_value = True
        
        mock_doc = Mock()
        mock_doc.id = "doc-1"
        mock_doc.name = "test.txt"
        mock_doc.content = "Test content"
        mock_doc.size_bytes = 100
        
        mock_processor = Mock()
        mock_processor.process.return_value = mock_doc
        mock_processor_class.return_value = mock_processor
        
        mock_chunk.return_value = [
            {"id": "c1", "text": "chunk 1", "document_id": "doc-1", "document_name": "test.txt"},
            {"id": "c2", "text": "chunk 2", "document_id": "doc-1", "document_name": "test.txt"},
        ]
        
        mock_emb_service = Mock()
        mock_emb_service.encode.return_value = Mock()
        mock_emb_service.dimension = 384
        mock_get_emb.return_value = mock_emb_service
        
        mock_store = Mock()
        mock_store.add.return_value = True
        mock_store.save.return_value = True
        mock_vector_store.return_value = mock_store
        
        engine = RAGEngine()
        success, message = engine.add_document("test.txt")
        
        assert success is True
        assert "Added" in message
        assert "doc-1" in engine.documents
    
    @patch('rag.engine.VectorStore')
    @patch('rag.engine.get_embedding_service')
    @patch('rag.engine.validate_file_extension')
    def test_add_invalid_extension(self, mock_validate_ext, mock_get_emb, mock_vector_store):
        """Test adding document with invalid extension."""
        mock_validate_ext.return_value = False
        
        mock_emb_service = Mock()
        mock_emb_service.dimension = 384
        mock_get_emb.return_value = mock_emb_service
        
        mock_store = Mock()
        mock_vector_store.return_value = mock_store
        
        engine = RAGEngine()
        success, message = engine.add_document("test.jpg")
        
        assert success is False
        assert "Invalid file type" in message
    
    @patch('rag.engine.VectorStore')
    @patch('rag.engine.get_embedding_service')
    @patch('rag.engine.validate_file_extension')
    @patch('rag.engine.validate_file_size')
    def test_add_oversized_file(self, mock_validate_size, mock_validate_ext, mock_get_emb, mock_vector_store):
        """Test adding oversized file."""
        mock_validate_ext.return_value = True
        mock_validate_size.return_value = False
        
        mock_emb_service = Mock()
        mock_emb_service.dimension = 384
        mock_get_emb.return_value = mock_emb_service
        
        mock_store = Mock()
        mock_vector_store.return_value = mock_store
        
        engine = RAGEngine()
        success, message = engine.add_document("large.txt")
        
        assert success is False
        assert "too large" in message.lower()


@pytest.mark.unit
class TestRAGEngineRetrieve:
    """Test document retrieval."""
    
    @patch('rag.engine.VectorStore')
    @patch('rag.engine.get_embedding_service')
    def test_retrieve_with_results(self, mock_get_emb, mock_vector_store):
        """Test retrieving documents."""
        # Setup mocks
        mock_emb_service = Mock()
        mock_emb_service.encode_query.return_value = Mock()
        mock_emb_service.dimension = 384
        mock_get_emb.return_value = mock_emb_service
        
        mock_store = Mock()
        mock_store.size = 5
        mock_store.search.return_value = [
            ({"chunk_id": "c1", "document_id": "d1", "document_name": "test.txt", "text": "chunk 1"}, 0.9),
            ({"chunk_id": "c2", "document_id": "d1", "document_name": "test.txt", "text": "chunk 2"}, 0.8),
        ]
        mock_vector_store.return_value = mock_store
        
        engine = RAGEngine()
        results = engine.retrieve("test query")
        
        assert len(results) == 2
        assert results[0]["score"] == 0.9
        assert results[0]["document_name"] == "test.txt"
    
    @patch('rag.engine.VectorStore')
    @patch('rag.engine.get_embedding_service')
    def test_retrieve_empty_store(self, mock_get_emb, mock_vector_store):
        """Test retrieving from empty store."""
        mock_emb_service = Mock()
        mock_emb_service.dimension = 384
        mock_get_emb.return_value = mock_emb_service
        
        mock_store = Mock()
        mock_store.size = 0
        mock_vector_store.return_value = mock_store
        
        engine = RAGEngine()
        results = engine.retrieve("test query")
        
        assert results == []


@pytest.mark.unit
class TestRAGEngineContextString:
    """Test context string generation."""
    
    @patch('rag.engine.VectorStore')
    @patch('rag.engine.get_embedding_service')
    @patch.object(RAGEngine, 'retrieve')
    def test_get_context_string(self, mock_retrieve, mock_get_emb, mock_vector_store):
        """Test getting context string."""
        mock_emb_service = Mock()
        mock_emb_service.dimension = 384
        mock_get_emb.return_value = mock_emb_service
        
        mock_store = Mock()
        mock_vector_store.return_value = mock_store
        
        mock_retrieve.return_value = [
            {"text": "Context chunk 1", "document_name": "doc1.txt", "score": 0.9},
            {"text": "Context chunk 2", "document_name": "doc2.txt", "score": 0.8},
        ]
        
        engine = RAGEngine()
        context = engine.get_context_string("query")
        
        assert "doc1.txt" in context
        assert "Context chunk 1" in context
        assert "doc2.txt" in context
    
    @patch('rag.engine.VectorStore')
    @patch('rag.engine.get_embedding_service')
    @patch.object(RAGEngine, 'retrieve')
    def test_get_context_string_empty(self, mock_retrieve, mock_get_emb, mock_vector_store):
        """Test getting context string with no results."""
        mock_emb_service = Mock()
        mock_emb_service.dimension = 384
        mock_get_emb.return_value = mock_emb_service
        
        mock_store = Mock()
        mock_vector_store.return_value = mock_store
        
        mock_retrieve.return_value = []
        
        engine = RAGEngine()
        context = engine.get_context_string("query")
        
        assert context == ""


@pytest.mark.unit
class TestRAGEngineDocumentList:
    """Test document list functionality."""
    
    @patch('rag.engine.VectorStore')
    @patch('rag.engine.get_embedding_service')
    def test_get_document_list(self, mock_get_emb, mock_vector_store):
        """Test getting document list."""
        mock_emb_service = Mock()
        mock_emb_service.dimension = 384
        mock_get_emb.return_value = mock_emb_service
        
        mock_store = Mock()
        mock_vector_store.return_value = mock_store
        
        engine = RAGEngine()
        
        # Add mock documents
        mock_doc1 = Mock()
        mock_doc1.id = "d1"
        mock_doc1.name = "doc1.txt"
        mock_doc1.type = "txt"
        mock_doc1.size_bytes = 100
        mock_doc1.chunks = [Mock(), Mock()]
        mock_doc1.upload_time = Mock()
        
        mock_doc2 = Mock()
        mock_doc2.id = "d2"
        mock_doc2.name = "doc2.txt"
        mock_doc2.type = "txt"
        mock_doc2.size_bytes = 200
        mock_doc2.chunks = [Mock()]
        mock_doc2.upload_time = Mock()
        
        engine.documents = {"d1": mock_doc1, "d2": mock_doc2}
        
        doc_list = engine.get_document_list()
        
        assert len(doc_list) == 2
        assert doc_list[0]["id"] in ["d1", "d2"]
        assert doc_list[0]["chunk_count"] in [1, 2]


@pytest.mark.unit
class TestRAGEngineClear:
    """Test clearing documents."""
    
    @patch('rag.engine.VectorStore')
    @patch('rag.engine.get_embedding_service')
    def test_clear_all(self, mock_get_emb, mock_vector_store):
        """Test clearing all documents."""
        mock_emb_service = Mock()
        mock_emb_service.dimension = 384
        mock_get_emb.return_value = mock_emb_service
        
        mock_store = Mock()
        mock_store.clear.return_value = None
        mock_store.save.return_value = True
        mock_vector_store.return_value = mock_store
        
        engine = RAGEngine()
        engine.documents = {"d1": Mock()}
        
        result = engine.clear_all()
        
        assert result is True
        assert engine.documents == {}


@pytest.mark.unit
class TestRAGEngineStats:
    """Test RAG engine statistics."""
    
    @patch('rag.engine.VectorStore')
    @patch('rag.engine.get_embedding_service')
    def test_get_stats(self, mock_get_emb, mock_vector_store):
        """Test getting engine statistics."""
        mock_emb_service = Mock()
        mock_emb_service.dimension = 384
        mock_get_emb.return_value = mock_emb_service
        
        mock_store = Mock()
        mock_store.get_stats.return_value = {
            "vector_count": 10,
            "document_count": 2,
            "dimension": 384,
        }
        mock_vector_store.return_value = mock_store
        
        engine = RAGEngine()
        engine.documents = {"d1": Mock(), "d2": Mock()}
        
        stats = engine.get_stats()
        
        assert stats["vector_count"] == 10
        assert stats["document_count"] == 2
        assert stats["documents"] == 2
        assert stats["dimension"] == 384
