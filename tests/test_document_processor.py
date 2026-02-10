"""Unit tests for rag.document_processor module."""

import os
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch

from rag.document_processor import (
    Document,
    Chunk,
    DocumentProcessor,
    process_file,
)


@pytest.mark.unit
class TestDocumentDataclass:
    """Test Document dataclass."""
    
    def test_document_creation(self):
        """Test Document creation."""
        doc = Document(
            id="doc-1",
            name="test.txt",
            type="txt",
            content="Test content",
            chunks=[],
            upload_time=datetime.now(),
            size_bytes=100,
        )
        
        assert doc.id == "doc-1"
        assert doc.name == "test.txt"
        assert doc.type == "txt"
        assert doc.content == "Test content"
        assert doc.size_bytes == 100
        assert doc.page_count is None
    
    def test_document_with_page_count(self):
        """Test Document with page count."""
        doc = Document(
            id="doc-2",
            name="test.pdf",
            type="pdf",
            content="PDF content",
            chunks=[],
            upload_time=datetime.now(),
            size_bytes=1000,
            page_count=5,
        )
        
        assert doc.page_count == 5


@pytest.mark.unit
class TestChunkDataclass:
    """Test Chunk dataclass."""
    
    def test_chunk_creation(self):
        """Test Chunk creation."""
        chunk = Chunk(
            id="chunk-1",
            text="Test chunk text",
            document_id="doc-1",
            document_name="test.txt",
        )
        
        assert chunk.id == "chunk-1"
        assert chunk.text == "Test chunk text"
        assert chunk.document_id == "doc-1"
        assert chunk.page_number is None
        assert chunk.embedding is None
    
    def test_chunk_with_embedding(self):
        """Test Chunk with embedding."""
        chunk = Chunk(
            id="chunk-2",
            text="Test text",
            document_id="doc-1",
            document_name="test.txt",
            embedding=[0.1, 0.2, 0.3],
        )
        
        assert chunk.embedding == [0.1, 0.2, 0.3]


@pytest.mark.unit
class TestDocumentProcessor:
    """Test DocumentProcessor class."""
    
    def test_initialization(self):
        """Test DocumentProcessor initialization."""
        processor = DocumentProcessor()
        
        assert processor.supported_types == {".txt", ".md", ".pdf"}
    
    def test_process_text_file(self, temp_dir):
        """Test processing a text file."""
        # Create test file
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("This is test content.")
        
        processor = DocumentProcessor()
        doc = processor.process(str(test_file))
        
        assert doc.name == "test.txt"
        assert doc.type == "txt"
        assert doc.content == "This is test content."
        assert doc.size_bytes > 0
        assert doc.id is not None
        assert doc.upload_time is not None
    
    def test_process_from_bytes(self, temp_dir):
        """Test processing from bytes."""
        test_file = Path(temp_dir) / "bytes_test.txt"
        content = b"Byte content test"
        
        processor = DocumentProcessor()
        doc = processor.process(str(test_file), file_content=content)
        
        assert doc.content == "Byte content test"
        assert doc.size_bytes == len(content)
    
    def test_unsupported_file_type(self, temp_dir):
        """Test processing unsupported file type."""
        test_file = Path(temp_dir) / "test.jpg"
        test_file.write_text("Not an image")
        
        processor = DocumentProcessor()
        
        with pytest.raises(ValueError) as exc_info:
            processor.process(str(test_file))
        
        assert "Unsupported file type" in str(exc_info.value)
    
    def test_read_text_file(self, temp_dir):
        """Test reading text file."""
        test_file = Path(temp_dir) / "read_test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3")
        
        processor = DocumentProcessor()
        content = processor._read_file(test_file)
        
        assert "Line 1" in content
        assert "Line 2" in content
        assert "Line 3" in content
    
    def test_read_markdown_file(self, temp_dir):
        """Test reading markdown file."""
        test_file = Path(temp_dir) / "test.md"
        test_file.write_text("# Heading\n\nSome **bold** text.")
        
        processor = DocumentProcessor()
        content = processor._read_file(test_file)
        
        assert "Heading" in content
        assert "bold" in content or "**bold**" in content


@pytest.mark.unit
class TestProcessFileFunction:
    """Test process_file convenience function."""
    
    def test_process_file(self, temp_dir):
        """Test process_file function."""
        test_file = Path(temp_dir) / "convenience.txt"
        test_file.write_text("Convenience test")
        
        doc = process_file(str(test_file))
        
        assert doc.name == "convenience.txt"
        assert doc.content == "Convenience test"


@pytest.mark.integration
class TestDocumentProcessingIntegration:
    """Integration tests for document processing."""
    
    def test_full_processing_pipeline(self, temp_dir):
        """Test complete document processing pipeline."""
        # Create test file with multiple paragraphs
        test_file = Path(temp_dir) / "integration.txt"
        test_file.write_text("""
        Paragraph 1: Introduction to the document.
        
        Paragraph 2: Main content with important information.
        
        Paragraph 3: Conclusion and summary.
        """)
        
        # Process document
        processor = DocumentProcessor()
        doc = processor.process(str(test_file))
        
        # Verify document structure
        assert doc.id is not None
        assert doc.name == "integration.txt"
        assert doc.type == "txt"
        assert len(doc.content) > 0
        assert doc.size_bytes == test_file.stat().st_size
        assert isinstance(doc.chunks, list)
    
    def test_large_file_processing(self, temp_dir):
        """Test processing large files."""
        test_file = Path(temp_dir) / "large.txt"
        # Create 1MB of content
        content = "Lorem ipsum dolor sit amet. " * 20000
        test_file.write_text(content)
        
        processor = DocumentProcessor()
        doc = processor.process(str(test_file))
        
        assert doc.size_bytes > 1000000  # > 1MB
        assert len(doc.content) > 0
