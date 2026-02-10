"""Unit tests for rag.vector_store module."""

import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from rag.vector_store import VectorStore


@pytest.mark.unit
class TestVectorStoreInitialization:
    """Test VectorStore initialization."""
    
    def test_default_initialization(self):
        """Test VectorStore initialization with defaults."""
        store = VectorStore()
        
        assert store.dimension == 384
        assert store._index is None  # Lazy loading
        assert store._metadata == {}
        assert store._id_counter == 0
        assert store._doc_count == 0
        assert store.last_updated is None
    
    def test_custom_dimension(self):
        """Test VectorStore with custom dimension."""
        store = VectorStore(dimension=768)
        
        assert store.dimension == 768
    
    def test_size_property(self):
        """Test size property."""
        store = VectorStore()
        
        assert store.size == 0
        
        # Simulate adding vectors
        store._id_counter = 5
        assert store.size == 5
    
    def test_doc_count_property(self):
        """Test doc_count property."""
        store = VectorStore()
        
        assert store.doc_count == 0
        
        # Simulate having documents
        store._doc_count = 3
        assert store.doc_count == 3


@pytest.mark.unit
class TestVectorStoreAdd:
    """Test adding vectors to store."""
    
    @patch('rag.vector_store.faiss')
    def test_add_vectors(self, mock_faiss):
        """Test adding vectors to store."""
        # Setup mock
        mock_index = Mock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.normalize_L2 = Mock()
        
        store = VectorStore(dimension=3)
        
        # Create test embeddings and metadata
        embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        metadata = [
            {"chunk_id": "c1", "document_id": "d1", "text": "text1"},
            {"chunk_id": "c2", "document_id": "d1", "text": "text2"},
        ]
        
        result = store.add(embeddings, metadata)
        
        assert result is True
        assert store._id_counter == 2
        assert store._doc_count == 1
        assert len(store._metadata) == 2
        assert store.last_updated is not None
    
    @patch('rag.vector_store.faiss')
    def test_add_empty_embeddings(self, mock_faiss):
        """Test adding empty embeddings."""
        store = VectorStore(dimension=3)
        
        result = store.add(np.array([]), [])
        
        assert result is True
        assert store._id_counter == 0
    
    @patch('rag.vector_store.faiss')
    def test_add_mismatched_lengths(self, mock_faiss):
        """Test adding mismatched embeddings and metadata."""
        store = VectorStore(dimension=3)
        
        embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        metadata = [
            {"chunk_id": "c1"},
            {"chunk_id": "c2"},
        ]
        
        with pytest.raises(ValueError) as exc_info:
            store.add(embeddings, metadata)
        
        assert "same length" in str(exc_info.value)
    
    @patch('rag.vector_store.faiss')
    def test_add_1d_embedding(self, mock_faiss):
        """Test adding 1D embedding."""
        mock_index = Mock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.normalize_L2 = Mock()
        
        store = VectorStore(dimension=3)
        
        # 1D array should be reshaped
        embeddings = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        metadata = [{"chunk_id": "c1"}]
        
        result = store.add(embeddings, metadata)
        
        assert result is True
        mock_index.add.assert_called_once()


@pytest.mark.unit
class TestVectorStoreSearch:
    """Test vector search functionality."""
    
    @patch('rag.vector_store.faiss')
    def test_search_vectors(self, mock_faiss):
        """Test searching vectors."""
        # Setup mock
        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8]]),  # scores
            np.array([[0, 1]])       # indices
        )
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.normalize_L2 = Mock()
        
        store = VectorStore(dimension=3)
        
        # Add some vectors first
        embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        metadata = [
            {"chunk_id": "c1", "document_id": "d1", "text": "text1"},
            {"chunk_id": "c2", "document_id": "d1", "text": "text2"},
        ]
        store.add(embeddings, metadata)
        
        # Search
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=2)
        
        assert len(results) == 2
        assert results[0][1] == 0.9  # score
        assert results[1][1] == 0.8
    
    def test_search_empty_store(self):
        """Test searching empty store."""
        store = VectorStore(dimension=3)
        
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=5)
        
        assert results == []
    
    @patch('rag.vector_store.faiss')
    def test_search_1d_query(self, mock_faiss):
        """Test searching with 1D query."""
        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[1.0]]),
            np.array([[0]])
        )
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.normalize_L2 = Mock()
        
        store = VectorStore(dimension=3)
        embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        metadata = [{"chunk_id": "c1"}]
        store.add(embeddings, metadata)
        
        # 1D query should be reshaped
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=1)
        
        assert len(results) == 1


@pytest.mark.unit
class TestVectorStoreRemove:
    """Test vector removal functionality."""
    
    @patch('rag.vector_store.faiss')
    def test_remove_by_document(self, mock_faiss):
        """Test removing vectors by document ID."""
        mock_faiss.normalize_L2 = Mock()
        
        store = VectorStore(dimension=3)
        
        # Add vectors from two documents
        embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        metadata = [
            {"chunk_id": "c1", "document_id": "d1"},
            {"chunk_id": "c2", "document_id": "d2"},
            {"chunk_id": "c3", "document_id": "d1"},
        ]
        store.add(embeddings, metadata)
        
        # Remove document d1
        removed = store.remove_by_document("d1")
        
        assert removed == 2
        assert store._doc_count == 1
    
    def test_remove_nonexistent_document(self):
        """Test removing non-existent document."""
        store = VectorStore(dimension=3)
        
        removed = store.remove_by_document("nonexistent")
        
        assert removed == 0


@pytest.mark.unit
class TestVectorStoreClear:
    """Test clearing vector store."""
    
    def test_clear_store(self):
        """Test clearing all vectors."""
        store = VectorStore(dimension=3)
        
        # Add some data
        store._metadata = {0: {"chunk_id": "c1"}, 1: {"chunk_id": "c2"}}
        store._id_counter = 2
        store._doc_count = 1
        
        store.clear()
        
        assert store._index is None
        assert store._metadata == {}
        assert store._id_counter == 0
        assert store._doc_count == 0
        assert store.last_updated is not None


@pytest.mark.unit
class TestVectorStoreStats:
    """Test vector store statistics."""
    
    def test_get_stats(self):
        """Test getting store statistics."""
        store = VectorStore(dimension=384)
        
        # Setup test data
        store._id_counter = 10
        store._doc_count = 3
        store.last_updated = datetime.now()
        
        stats = store.get_stats()
        
        assert stats["vector_count"] == 10
        assert stats["document_count"] == 3
        assert stats["dimension"] == 384
        assert stats["last_updated"] == store.last_updated


@pytest.mark.error_handling
class TestVectorStoreErrors:
    """Test VectorStore error handling."""
    
    @patch('rag.vector_store.faiss')
    def test_add_error(self, mock_faiss):
        """Test handling add errors."""
        mock_index = Mock()
        mock_index.add.side_effect = RuntimeError("FAISS error")
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.normalize_L2 = Mock()
        
        store = VectorStore(dimension=3)
        embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        metadata = [{"chunk_id": "c1"}]
        
        result = store.add(embeddings, metadata)
        
        assert result is False
    
    @patch('rag.vector_store.faiss')
    def test_search_error(self, mock_faiss):
        """Test handling search errors."""
        mock_index = Mock()
        mock_index.search.side_effect = RuntimeError("Search error")
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.normalize_L2 = Mock()
        
        store = VectorStore(dimension=3)
        store._index = mock_index
        store._id_counter = 1
        
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=5)
        
        assert results == []
