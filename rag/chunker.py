"""Text chunking strategies."""

import re
from dataclasses import dataclass
from typing import List, Optional
from uuid import uuid4

from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class TextChunk:
    """Represents a text chunk with metadata."""
    id: str
    text: str
    start_pos: int
    end_pos: int
    word_count: int


class TextChunker:
    """Split text into semantic chunks."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separator: str = "\n\n"
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.separator = separator
    
    def chunk(self, text: str, document_id: str = "", document_name: str = "") -> List[dict]:
        """Split text into chunks.
        
        Args:
            text: Text to chunk.
            document_id: Document identifier.
            document_name: Document name.
            
        Returns:
            List of chunk dictionaries.
        """
        if not text or not text.strip():
            return []
        
        # First, try to split by paragraphs
        paragraphs = self._split_by_paragraphs(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph_size = len(paragraph)
            
            # If adding this paragraph would exceed chunk size, finalize current chunk
            if current_size + paragraph_size > self.chunk_size and current_chunk:
                chunk_text = self.separator.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, document_id, document_name))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = [overlap_text, paragraph] if overlap_text else [paragraph]
                current_size = len(overlap_text) + paragraph_size if overlap_text else paragraph_size
            else:
                current_chunk.append(paragraph)
                current_size += paragraph_size + len(self.separator)
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = self.separator.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, document_id, document_name))
        
        logger.debug(f"Created {len(chunks)} chunks from {len(text)} characters")
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        
        # Split by multiple newlines or section markers
        paragraphs = re.split(r'\n\s*\n|\n\s*---\s*Page \d+\s*---\s*\n', text)
        
        # Clean and filter
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Further split very long paragraphs
        result = []
        for para in paragraphs:
            if len(para) > self.chunk_size:
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) > self.chunk_size and current:
                        result.append(current.strip())
                        current = sent
                    else:
                        current = f"{current} {sent}".strip()
                if current:
                    result.append(current)
            else:
                result.append(para)
        
        return result
    
    def _get_overlap(self, chunks: List[str]) -> str:
        """Get overlapping text from previous chunks."""
        overlap_text = ""
        total_size = 0
        
        # Go backwards through chunks to get overlap
        for chunk in reversed(chunks):
            if total_size + len(chunk) <= self.chunk_overlap:
                overlap_text = chunk + self.separator + overlap_text
                total_size += len(chunk) + len(self.separator)
            else:
                # Take partial chunk
                remaining = self.chunk_overlap - total_size
                if remaining > 50:  # Minimum meaningful overlap
                    overlap_text = chunk[-remaining:] + self.separator + overlap_text
                break
        
        return overlap_text.strip()
    
    def _create_chunk(self, text: str, document_id: str, document_name: str) -> dict:
        """Create a chunk dictionary."""
        return {
            "id": str(uuid4()),
            "text": text.strip(),
            "document_id": document_id,
            "document_name": document_name,
            "page_number": None,
        }


class RecursiveCharacterChunker(TextChunker):
    """Chunk text using recursive character splitting."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    
    def chunk(self, text: str, document_id: str = "", document_name: str = "") -> List[dict]:
        """Recursively split text by separators."""
        if not text or not text.strip():
            return []
        
        chunks = self._recursive_split(text, 0)
        
        result = []
        for chunk_text in chunks:
            if chunk_text.strip():
                result.append(self._create_chunk(chunk_text.strip(), document_id, document_name))
        
        return result
    
    def _recursive_split(self, text: str, separator_idx: int) -> List[str]:
        """Recursively split text."""
        if separator_idx >= len(self.separators):
            # Last resort: split by character
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        
        separator = self.separators[separator_idx]
        
        if separator == "":
            # Split by character
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
        
        # Split by current separator
        if separator in text:
            parts = text.split(separator)
        else:
            return self._recursive_split(text, separator_idx + 1)
        
        # Merge small parts
        result = []
        current = ""
        
        for part in parts:
            test = current + separator + part if current else part
            
            if len(test) <= self.chunk_size:
                current = test
            else:
                if current:
                    result.append(current)
                    # Add overlap
                    overlap_start = max(0, len(current) - self.chunk_overlap)
                    current = current[overlap_start:] + separator + part
                else:
                    # Part is too long, recurse
                    result.extend(self._recursive_split(part, separator_idx + 1))
                    current = ""
        
        if current:
            result.append(current)
        
        return result


def chunk_text(
    text: str,
    document_id: str = "",
    document_name: str = "",
    chunk_size: int = None,
    chunk_overlap: int = None,
    use_recursive: bool = True
) -> List[dict]:
    """Convenience function to chunk text.
    
    Args:
        text: Text to chunk.
        document_id: Document identifier.
        document_name: Document name.
        chunk_size: Size of each chunk.
        chunk_overlap: Overlap between chunks.
        use_recursive: Use recursive chunking strategy.
        
    Returns:
        List of chunk dictionaries.
    """
    if use_recursive:
        chunker = RecursiveCharacterChunker(chunk_size, chunk_overlap)
    else:
        chunker = TextChunker(chunk_size, chunk_overlap)
    
    return chunker.chunk(text, document_id, document_name)
