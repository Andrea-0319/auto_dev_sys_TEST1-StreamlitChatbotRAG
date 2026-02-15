"""FAISS vector store for similarity search."""

import json
import pickle
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class VectorStore:
    """FAISS-based vector store with metadata."""
    
    def __init__(self, dimension: int = 384):
        """Initialize vector store.
        
        Args:
            dimension: Embedding dimension.
        """
        self.dimension = dimension
        self._index = None
        self._metadata: Dict[int, dict] = {}
        self._id_counter = 0
        self._doc_count = 0
        self.last_updated: Optional[datetime] = None
    
    def _get_index(self):
        """Lazy initialize FAISS index."""
        if self._index is None:
            try:
                import faiss
                # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
                self._index = faiss.IndexFlatIP(self.dimension)
                logger.info(f"Initialized FAISS index with dimension {self.dimension}")
            except ImportError:
                logger.error("FAISS not installed. Please install with: pip install faiss-cpu")
                raise
        return self._index
    
    def add(self, embeddings: np.ndarray, metadata_list: List[dict]) -> bool:
        """Add embeddings with metadata to the store.
        
        Args:
            embeddings: Array of embeddings (n_vectors x dimension).
            metadata_list: List of metadata dicts for each embedding.
            
        Returns:
            True if successful.
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Embeddings and metadata must have same length")
        
        if len(embeddings) == 0:
            return True
        
        try:
            index = self._get_index()
            
            # Ensure correct shape and type
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            embeddings = embeddings.astype(np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            start_id = self._id_counter
            index.add(embeddings)
            
            # Store metadata
            for i, meta in enumerate(metadata_list):
                self._metadata[start_id + i] = meta
            
            self._id_counter += len(embeddings)
            self._doc_count += len(set(m.get("document_id") for m in metadata_list))
            self.last_updated = datetime.now()
            
            logger.info(f"Added {len(embeddings)} vectors to store. Total: {self._id_counter}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding vectors to store: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[dict, float]]:
        """Search for similar vectors.
        
        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            
        Returns:
            List of (metadata, score) tuples.
        """
        if self._index is None or self._id_counter == 0:
            return []
        
        try:
            # Ensure correct shape
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            query_embedding = query_embedding.astype(np.float32)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self._index.search(query_embedding, min(top_k, self._id_counter))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx in self._metadata:
                    results.append((self._metadata[idx], float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []
    
    def remove_by_document(self, document_id: str) -> int:
        """Remove all vectors for a document.
        
        Args:
            document_id: Document ID to remove.
            
        Returns:
            Number of vectors removed.
        """
        # FAISS doesn't support direct removal, so we need to rebuild
        indices_to_remove = [
            idx for idx, meta in self._metadata.items()
            if meta.get("document_id") == document_id
        ]
        
        if not indices_to_remove:
            return 0
        
        # Rebuild index without removed vectors
        try:
            import faiss
            
            # Get all remaining embeddings and metadata
            remaining_embeddings = []
            remaining_metadata = {}
            new_idx = 0
            
            for idx, meta in self._metadata.items():
                if idx not in indices_to_remove:
                    # Get embedding from old index
                    # Note: This is inefficient but necessary since FAISS IndexFlatIP doesn't support remove
                    # For production, consider using IndexIDMap or other structures
                    remaining_embeddings.append(None)  # Placeholder
                    remaining_metadata[new_idx] = meta
                    new_idx += 1
            
            # For now, just remove from metadata
            for idx in indices_to_remove:
                del self._metadata[idx]
            
            self._doc_count = len(set(m.get("document_id") for m in self._metadata.values()))
            self.last_updated = datetime.now()
            
            logger.info(f"Removed document {document_id} ({len(indices_to_remove)} vectors)")
            return len(indices_to_remove)
            
        except Exception as e:
            logger.error(f"Error removing vectors: {e}")
            return 0
    
    def clear(self) -> None:
        """Clear all vectors and metadata."""
        self._index = None
        self._metadata = {}
        self._id_counter = 0
        self._doc_count = 0
        self.last_updated = datetime.now()
        logger.info("Vector store cleared")
    
    def save(self, directory: str = None) -> bool:
        """Save vector store to disk.
        
        Args:
            directory: Directory to save to.
            
        Returns:
            True if successful.
        """
        if directory is None:
            directory = settings.VECTOR_STORE_DIR
        
        try:
            dir_path = Path(directory)
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            if self._index is not None:
                import faiss
                faiss.write_index(self._index, str(dir_path / "index.faiss"))
            
            # Save metadata
            with open(dir_path / "metadata.json", "w") as f:
                json.dump({
                    "metadata": self._metadata,
                    "id_counter": self._id_counter,
                    "doc_count": self._doc_count,
                    "dimension": self.dimension,
                    "last_updated": self.last_updated.isoformat() if self.last_updated else None
                }, f)
            
            logger.info(f"Vector store saved to {directory}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            return False
    
    def load(self, directory: str = None) -> bool:
        """Load vector store from disk.
        
        Args:
            directory: Directory to load from.
            
        Returns:
            True if successful.
        """
        if directory is None:
            directory = settings.VECTOR_STORE_DIR
        
        try:
            dir_path = Path(directory)
            
            if not (dir_path / "index.faiss").exists():
                logger.warning(f"No saved vector store found at {directory}")
                return False
            
            # Load FAISS index
            import faiss
            self._index = faiss.read_index(str(dir_path / "index.faiss"))
            
            # Load metadata
            with open(dir_path / "metadata.json", "r") as f:
                data = json.load(f)
                self._metadata = {int(k): v for k, v in data["metadata"].items()}
                self._id_counter = data["id_counter"]
                self._doc_count = data["doc_count"]
                self.dimension = data["dimension"]
                if data.get("last_updated"):
                    self.last_updated = datetime.fromisoformat(data["last_updated"])
            
            logger.info(f"Vector store loaded from {directory}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    @property
    def size(self) -> int:
        """Get number of vectors in store."""
        return self._id_counter
    
    @property
    def doc_count(self) -> int:
        """Get number of documents."""
        return self._doc_count
    
    def get_stats(self) -> dict:
        """Get store statistics."""
        return {
            "vector_count": self.size,
            "document_count": self.doc_count,
            "dimension": self.dimension,
            "last_updated": self.last_updated
        }
