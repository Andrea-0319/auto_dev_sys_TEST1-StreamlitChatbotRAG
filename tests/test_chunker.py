"""Unit tests for rag.chunker module."""

import pytest
from rag.chunker import (
    TextChunk,
    TextChunker,
    RecursiveCharacterChunker,
    chunk_text,
)


@pytest.mark.unit
class TestTextChunkDataclass:
    """Test TextChunk dataclass."""
    
    def test_chunk_creation(self):
        """Test TextChunk creation."""
        chunk = TextChunk(
            id="chunk-1",
            text="Test text",
            start_pos=0,
            end_pos=9,
            word_count=2,
        )
        
        assert chunk.id == "chunk-1"
        assert chunk.text == "Test text"
        assert chunk.start_pos == 0
        assert chunk.end_pos == 9
        assert chunk.word_count == 2


@pytest.mark.unit
class TestTextChunker:
    """Test TextChunker class."""
    
    def test_initialization_defaults(self):
        """Test TextChunker initialization with defaults."""
        chunker = TextChunker()
        
        assert chunker.chunk_size == 512  # Default from settings
        assert chunker.chunk_overlap == 50  # Default from settings
        assert chunker.separator == "\n\n"
    
    def test_initialization_custom(self):
        """Test TextChunker initialization with custom values."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10, separator="\n")
        
        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 10
        assert chunker.separator == "\n"
    
    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunker = TextChunker(chunk_size=100)
        chunks = chunker.chunk("")
        
        assert chunks == []
    
    def test_chunk_whitespace_only(self):
        """Test chunking whitespace-only text."""
        chunker = TextChunker(chunk_size=100)
        chunks = chunker.chunk("   \n\n\t  ")
        
        assert chunks == []
    
    def test_chunk_small_text(self):
        """Test chunking text smaller than chunk size."""
        chunker = TextChunker(chunk_size=100)
        text = "This is a short text."
        chunks = chunker.chunk(text, document_id="doc-1", document_name="test.txt")
        
        assert len(chunks) == 1
        assert chunks[0]["text"] == text
        assert chunks[0]["document_id"] == "doc-1"
        assert chunks[0]["document_name"] == "test.txt"
        assert "id" in chunks[0]
    
    def test_chunk_large_text(self):
        """Test chunking text larger than chunk size."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        # Create text with multiple paragraphs
        text = "\n\n".join([f"Paragraph {i} with some content here." for i in range(10)])
        
        chunks = chunker.chunk(text, document_id="doc-1", document_name="test.txt")
        
        assert len(chunks) > 1
        # Check that all chunks have required fields
        for chunk in chunks:
            assert "id" in chunk
            assert "text" in chunk
            assert "document_id" in chunk
            assert "document_name" in chunk
    
    def test_chunk_overlap(self):
        """Test that chunks have overlap."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        
        chunks = chunker.chunk(text, document_id="doc-1", document_name="test.txt")
        
        if len(chunks) > 1:
            # Check for overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                chunk1_text = chunks[i]["text"]
                chunk2_text = chunks[i + 1]["text"]
                # There should be some overlap
                assert any(word in chunk2_text for word in chunk1_text.split()[:3])
    
    def test_split_by_paragraphs(self):
        """Test paragraph splitting."""
        chunker = TextChunker()
        text = "Para 1\n\nPara 2\n\nPara 3"
        
        paragraphs = chunker._split_by_paragraphs(text)
        
        assert len(paragraphs) == 3
        assert "Para 1" in paragraphs[0]
        assert "Para 2" in paragraphs[1]
        assert "Para 3" in paragraphs[2]
    
    def test_split_by_page_markers(self):
        """Test splitting by page markers."""
        chunker = TextChunker()
        text = "Page 1 content\n\n--- Page 2 ---\n\nPage 2 content"
        
        paragraphs = chunker._split_by_paragraphs(text)
        
        assert "Page 1 content" in paragraphs[0]
        assert "Page 2 content" in paragraphs[1]
    
    def test_long_paragraph_sentence_split(self):
        """Test that long paragraphs are split by sentences."""
        chunker = TextChunker(chunk_size=50)
        # Create a very long single paragraph
        text = " ".join([f"Sentence number {i} in this long paragraph." for i in range(20)])
        
        paragraphs = chunker._split_by_paragraphs(text)
        
        # Should be split into multiple parts
        assert len(paragraphs) > 1


@pytest.mark.unit
class TestRecursiveCharacterChunker:
    """Test RecursiveCharacterChunker class."""
    
    def test_initialization(self):
        """Test RecursiveCharacterChunker initialization."""
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
        
        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 10
        assert chunker.separators == ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    
    def test_recursive_split_empty(self):
        """Test recursive split with empty text."""
        chunker = RecursiveCharacterChunker(chunk_size=100)
        chunks = chunker.chunk("")
        
        assert chunks == []
    
    def test_recursive_split_small_text(self):
        """Test recursive split with small text."""
        chunker = RecursiveCharacterChunker(chunk_size=100)
        text = "Small text"
        chunks = chunker.chunk(text, document_id="doc-1", document_name="test.txt")
        
        assert len(chunks) == 1
        assert chunks[0]["text"] == text
    
    def test_recursive_split_by_paragraphs(self):
        """Test recursive split respects paragraph boundaries."""
        chunker = RecursiveCharacterChunker(chunk_size=200)
        text = "First paragraph.\n\nSecond paragraph with more content.\n\nThird paragraph."
        
        chunks = chunker.chunk(text, document_id="doc-1", document_name="test.txt")
        
        # Should create reasonable chunks
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk["text"]) <= chunker.chunk_size


@pytest.mark.unit
class TestChunkTextFunction:
    """Test chunk_text convenience function."""
    
    def test_chunk_text_recursive(self):
        """Test chunk_text with recursive chunker."""
        text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        chunks = chunk_text(
            text,
            document_id="doc-1",
            document_name="test.txt",
            use_recursive=True
        )
        
        assert len(chunks) > 0
        assert all("id" in chunk for chunk in chunks)
    
    def test_chunk_text_standard(self):
        """Test chunk_text with standard chunker."""
        text = "Para 1\n\nPara 2\n\nPara 3"
        chunks = chunk_text(
            text,
            document_id="doc-1",
            document_name="test.txt",
            use_recursive=False
        )
        
        assert len(chunks) > 0
        assert all("id" in chunk for chunk in chunks)
    
    def test_chunk_text_custom_size(self):
        """Test chunk_text with custom chunk size."""
        text = " ".join([f"Word{i}" for i in range(100)])
        chunks = chunk_text(
            text,
            document_id="doc-1",
            document_name="test.txt",
            chunk_size=50,
            use_recursive=True
        )
        
        assert len(chunks) > 1  # Should create multiple chunks


@pytest.mark.edge_case
class TestChunkerEdgeCases:
    """Test chunker edge cases."""
    
    def test_very_long_paragraph(self):
        """Test chunking very long paragraph."""
        chunker = TextChunker(chunk_size=100)
        text = "word " * 1000  # Very long text
        
        chunks = chunker.chunk(text, document_id="doc-1", document_name="test.txt")
        
        assert len(chunks) > 0
        # All chunks should be within size limit
        for chunk in chunks:
            assert len(chunk["text"]) <= chunker.chunk_size + chunker.chunk_overlap
    
    def test_single_sentence(self):
        """Test chunking single sentence."""
        chunker = TextChunker(chunk_size=100)
        text = "This is a single sentence."
        
        chunks = chunker.chunk(text, document_id="doc-1", document_name="test.txt")
        
        assert len(chunks) == 1
        assert chunks[0]["text"] == text
    
    def test_only_newlines(self):
        """Test chunking text with only newlines."""
        chunker = TextChunker(chunk_size=100)
        text = "\n\n\n\n\n"
        
        chunks = chunker.chunk(text, document_id="doc-1", document_name="test.txt")
        
        assert chunks == []
    
    def test_unicode_text(self):
        """Test chunking unicode text."""
        chunker = TextChunker(chunk_size=100)
        text = "こんにちは世界 🌍 ñoño émojis: 🚀🔥"
        
        chunks = chunker.chunk(text, document_id="doc-1", document_name="test.txt")
        
        assert len(chunks) == 1
        assert "🌍" in chunks[0]["text"]
