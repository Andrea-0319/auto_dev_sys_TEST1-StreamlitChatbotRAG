"""Main RAG engine orchestration."""

from pathlib import Path
from typing import List, Optional, Tuple

from config import settings
from utils.logger import setup_logger
from utils.validators import validate_file_extension, validate_file_size
from rag.document_processor import DocumentProcessor, process_file
from rag.chunker import chunk_text
from rag.embeddings import get_embedding_service
from rag.vector_store import VectorStore

logger = setup_logger(__name__)


class RAGEngine:
    """Main RAG orchestration engine."""
    
    def __init__(self):
        """Initialize RAG engine."""
        self.doc_processor = DocumentProcessor()
        self.embedding_service = get_embedding_service()
        self.vector_store = VectorStore(dimension=self.embedding_service.dimension)
        self.documents: dict = {}
        
        # Try to load existing vector store
        self.vector_store.load()
    
    def add_document(self, file_path: str, file_content: bytes = None) -> Tuple[bool, str]:
        """Process and add a document to the knowledge base.
        
        Args:
            file_path: Path to the file.
            file_content: Optional file content bytes.
            
        Returns:
            Tuple of (success: bool, message: str).
        """
        try:
            path = Path(file_path)
            
            # Validate file type
            if not validate_file_extension(path.name):
                return False, f"Invalid file type. Allowed: {settings.ALLOWED_EXTENSIONS}"
            
            # Validate file size
            if file_content:
                file_size = len(file_content)
            else:
                file_size = path.stat().st_size
            
            if not validate_file_size(file_size):
                return False, f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
            
            # Process document
            logger.info(f"Processing document: {path.name}")
            document = self.doc_processor.process(file_path, file_content)
            
            # Check for duplicate
            if document.id in self.documents:
                return False, "Document already exists"
            
            # Chunk document
            logger.info(f"Chunking document: {document.name}")
            chunks = chunk_text(
                document.content,
                document_id=document.id,
                document_name=document.name,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            
            if not chunks:
                return False, "No text content found in document"
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_service.encode(texts, show_progress=True)
            
            # Add to vector store
            metadata_list = [
                {
                    "chunk_id": chunk["id"],
                    "document_id": chunk["document_id"],
                    "document_name": chunk["document_name"],
                    "text": chunk["text"],
                    "page_number": chunk.get("page_number")
                }
                for chunk in chunks
            ]
            
            success = self.vector_store.add(embeddings, metadata_list)
            
            if not success:
                return False, "Failed to add to vector store"
            
            # Store document info
            document.chunks = chunks
            self.documents[document.id] = document
            
            # Save vector store
            self.vector_store.save()
            
            logger.info(f"Successfully added document: {document.name}")
            return True, f"Added '{document.name}' ({len(chunks)} chunks)"
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False, f"Error: {str(e)}"
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the knowledge base.
        
        Args:
            doc_id: Document ID.
            
        Returns:
            True if removed successfully.
        """
        if doc_id not in self.documents:
            return False
        
        try:
            document = self.documents[doc_id]
            
            # Remove from vector store
            self.vector_store.remove_by_document(doc_id)
            
            # Remove from documents dict
            del self.documents[doc_id]
            
            # Save vector store
            self.vector_store.save()
            
            logger.info(f"Removed document: {document.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing document: {e}")
            return False
    
    def retrieve(self, query: str, top_k: int = None) -> List[dict]:
        """Retrieve relevant chunks for a query.
        
        Args:
            query: Search query.
            top_k: Number of results to return.
            
        Returns:
            List of chunk dictionaries with similarity scores.
        """
        if top_k is None:
            top_k = settings.TOP_K_RETRIEVAL
        
        if self.vector_store.size == 0:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.encode_query(query)
            
            # Search
            results = self.vector_store.search(query_embedding, top_k=top_k)
            
            # Format results
            chunks = []
            for metadata, score in results:
                chunks.append({
                    "id": metadata["chunk_id"],
                    "text": metadata["text"],
                    "document_id": metadata["document_id"],
                    "document_name": metadata["document_name"],
                    "page_number": metadata.get("page_number"),
                    "score": score
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def get_context_string(self, query: str, max_tokens: int = None) -> str:
        """Get formatted context string for model prompting.
        
        Args:
            query: Search query.
            max_tokens: Maximum context length.
            
        Returns:
            Concatenated relevant chunk texts.
        """
        if max_tokens is None:
            max_tokens = settings.MAX_CONTEXT_TOKENS
        
        chunks = self.retrieve(query)
        
        if not chunks:
            return ""
        
        # Build context string
        context_parts = []
        current_length = 0
        
        for chunk in chunks:
            chunk_text = f"[Source: {chunk['document_name']}]: {chunk['text']}\n\n"
            chunk_length = len(chunk_text)
            
            if current_length + chunk_length > max_tokens:
                break
            
            context_parts.append(chunk_text)
            current_length += chunk_length
        
        return "".join(context_parts).strip()
    
    def get_document_list(self) -> List[dict]:
        """Get list of all documents.
        
        Returns:
            List of document info dictionaries.
        """
        return [
            {
                "id": doc.id,
                "name": doc.name,
                "type": doc.type,
                "size_bytes": doc.size_bytes,
                "chunk_count": len(doc.chunks),
                "upload_time": doc.upload_time
            }
            for doc in self.documents.values()
        ]
    
    def clear_all(self) -> bool:
        """Clear all documents and reset the store.
        
        Returns:
            True if successful.
        """
        try:
            self.vector_store.clear()
            self.documents = {}
            self.vector_store.save()
            logger.info("Cleared all documents")
            return True
        except Exception as e:
            logger.error(f"Error clearing store: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get engine statistics.
        
        Returns:
            Dictionary with statistics.
        """
        store_stats = self.vector_store.get_stats()
        return {
            **store_stats,
            "documents": len(self.documents)
        }
