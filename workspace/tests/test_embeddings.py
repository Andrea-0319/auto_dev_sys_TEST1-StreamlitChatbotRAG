"""Unit tests for rag.embeddings module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from rag.embeddings import EmbeddingService, get_embedding_service


@pytest.mark.unit
class TestEmbeddingService:
    """Test EmbeddingService class."""
    
    def test_initialization_defaults(self):
        """Test EmbeddingService initialization with defaults."""
        service = EmbeddingService()
        
        assert service.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert service._model is None  # Lazy loading
        assert service._dimension == 384
    
    def test_initialization_custom(self):
        """Test EmbeddingService initialization with custom model."""
        service = EmbeddingService(model_name="custom-model")
        
        assert service.model_name == "custom-model"
        assert service._model is None
    
    @patch('rag.embeddings.SentenceTransformer')
    def test_load_model(self, mock_transformer_class):
        """Test lazy model loading."""
        # Setup mock
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer_class.return_value = mock_model
        
        service = EmbeddingService()
        
        # Model should be None initially
        assert service._model is None
        
        # Load model
        service._load_model()
        
        # Model should be loaded now
        assert service._model is not None
        mock_transformer_class.assert_called_once()
    
    @patch('rag.embeddings.SentenceTransformer')
    def test_load_model_once(self, mock_transformer_class):
        """Test that model is loaded only once."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer_class.return_value = mock_model
        
        service = EmbeddingService()
        
        # Load model twice
        service._load_model()
        service._load_model()
        
        # Should only be called once
        mock_transformer_class.assert_called_once()
    
    @patch('rag.embeddings.SentenceTransformer')
    def test_encode_texts(self, mock_transformer_class):
        """Test encoding texts."""
        # Setup mock
        mock_model = Mock()
        mock_embeddings = np.random.randn(2, 384).astype(np.float32)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer_class.return_value = mock_model
        
        service = EmbeddingService()
        texts = ["Hello world", "Test sentence"]
        result = service.encode(texts)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 384)
        mock_model.encode.assert_called_once()
    
    @patch('rag.embeddings.SentenceTransformer')
    def test_encode_empty_list(self, mock_transformer_class):
        """Test encoding empty list."""
        service = EmbeddingService()
        result = service.encode([])
        
        assert isinstance(result, np.ndarray)
        assert result.size == 0
    
    @patch('rag.embeddings.SentenceTransformer')
    def test_encode_query(self, mock_transformer_class):
        """Test encoding a single query."""
        # Setup mock
        mock_model = Mock()
        mock_embedding = np.random.randn(384).astype(np.float32)
        mock_model.encode.return_value = np.array([mock_embedding])
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer_class.return_value = mock_model
        
        service = EmbeddingService()
        result = service.encode_query("Test query")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)
    
    @patch('rag.embeddings.SentenceTransformer')
    def test_dimension_property(self, mock_transformer_class):
        """Test dimension property triggers model loading."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_transformer_class.return_value = mock_model
        
        service = EmbeddingService()
        dim = service.dimension
        
        assert dim == 768
        mock_transformer_class.assert_called_once()
    
    def test_similarity(self):
        """Test similarity calculation."""
        service = EmbeddingService()
        
        # Create normalized vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        vec3 = np.array([0.0, 1.0, 0.0])
        
        # Same vectors should have similarity 1
        sim1 = service.similarity(vec1, vec2)
        assert abs(sim1 - 1.0) < 0.001
        
        # Orthogonal vectors should have similarity 0
        sim2 = service.similarity(vec1, vec3)
        assert abs(sim2 - 0.0) < 0.001


@pytest.mark.unit
class TestGetEmbeddingService:
    """Test get_embedding_service function."""
    
    def test_singleton_creation(self):
        """Test that singleton is created."""
        # Reset singleton
        import rag.embeddings as emb_module
        emb_module._embedding_service = None
        
        service1 = get_embedding_service()
        service2 = get_embedding_service()
        
        assert service1 is service2
    
    def test_singleton_persistence(self):
        """Test that singleton persists across calls."""
        service1 = get_embedding_service()
        service2 = get_embedding_service()
        service3 = get_embedding_service()
        
        assert service1 is service2 is service3


@pytest.mark.error_handling
class TestEmbeddingServiceErrors:
    """Test EmbeddingService error handling."""
    
    @patch('rag.embeddings.SentenceTransformer')
    def test_load_model_failure(self, mock_transformer_class):
        """Test handling model loading failure."""
        mock_transformer_class.side_effect = ImportError("Model not found")
        
        service = EmbeddingService()
        
        with pytest.raises(ImportError):
            service._load_model()
    
    @patch('rag.embeddings.SentenceTransformer')
    def test_encode_failure(self, mock_transformer_class):
        """Test handling encoding failure."""
        mock_model = Mock()
        mock_model.encode.side_effect = RuntimeError("Encoding failed")
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer_class.return_value = mock_model
        
        service = EmbeddingService()
        
        with pytest.raises(RuntimeError):
            service.encode(["Test text"])
