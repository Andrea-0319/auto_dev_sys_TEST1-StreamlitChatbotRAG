"""Embedding generation service."""

from typing import List, Optional
import numpy as np

from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingService:
    """Generate embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = None):
        """Initialize embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model.
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None
        self._dimension = 384  # Default for all-MiniLM-L6-v2
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Embedding model loaded. Dimension: {self._dimension}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Generate embeddings for texts.
        
        Args:
            texts: List of texts to encode.
            batch_size: Batch size for encoding.
            show_progress: Show progress bar.
            
        Returns:
            Numpy array of embeddings.
        """
        self._load_model()
        
        if not texts:
            return np.array([])
        
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def encode_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query.
        
        Args:
            query: Query text.
            
        Returns:
            Embedding vector.
        """
        embeddings = self.encode([query])
        return embeddings[0] if len(embeddings) > 0 else None
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        self._load_model()
        return self._dimension
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding.
            embedding2: Second embedding.
            
        Returns:
            Cosine similarity score.
        """
        return float(np.dot(embedding1, embedding2))


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
