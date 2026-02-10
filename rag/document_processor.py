"""Document processing for various file types."""

import io
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union
from uuid import uuid4

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Document:
    """Represents a processed document."""
    id: str
    name: str
    type: str
    content: str
    chunks: List["Chunk"]
    upload_time: datetime
    size_bytes: int
    page_count: Optional[int] = None


@dataclass
class Chunk:
    """Represents a text chunk."""
    id: str
    text: str
    document_id: str
    document_name: str
    page_number: Optional[int] = None
    embedding: Optional[List[float]] = None


class DocumentProcessor:
    """Process various document formats."""
    
    def __init__(self):
        self.supported_types = {".txt", ".md", ".pdf"}
    
    def process(self, file_path: Union[str, Path], file_content: Optional[bytes] = None) -> Document:
        """Process a document file.
        
        Args:
            file_path: Path to the file.
            file_content: Optional file content bytes.
            
        Returns:
            Processed Document object.
        """
        path = Path(file_path)
        file_type = path.suffix.lower()
        
        if file_type not in self.supported_types:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Read content
        if file_content is not None:
            content = self._read_content(file_type, file_content)
            size_bytes = len(file_content)
        else:
            content = self._read_file(path)
            size_bytes = path.stat().st_size
        
        # Create document
        doc = Document(
            id=str(uuid4()),
            name=path.name,
            type=file_type.lstrip("."),
            content=content,
            chunks=[],
            upload_time=datetime.now(),
            size_bytes=size_bytes,
        )
        
        logger.info(f"Processed document: {doc.name} ({size_bytes} bytes)")
        return doc
    
    def _read_file(self, path: Path) -> str:
        """Read file content based on type."""
        file_type = path.suffix.lower()
        
        if file_type == ".pdf":
            return self._read_pdf(path)
        elif file_type == ".md":
            return self._read_markdown(path)
        else:  # .txt
            return self._read_text(path)
    
    def _read_content(self, file_type: str, content: bytes) -> str:
        """Read content from bytes."""
        if file_type == ".pdf":
            return self._read_pdf_bytes(content)
        elif file_type == ".md":
            return content.decode("utf-8", errors="ignore")
        else:  # .txt
            return content.decode("utf-8", errors="ignore")
    
    def _read_text(self, path: Path) -> str:
        """Read text file."""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    
    def _read_pdf(self, path: Path) -> str:
        """Read PDF file."""
        if not PYPDF2_AVAILABLE:
            logger.warning("PyPDF2 not available, trying basic text extraction")
            try:
                with open(path, "rb") as f:
                    return self._read_pdf_bytes(f.read())
            except Exception as e:
                raise ImportError(f"PyPDF2 is required for PDF processing: {e}")
        
        text = ""
        try:
            with open(path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
        except Exception as e:
            logger.error(f"Error reading PDF {path}: {e}")
            raise
        
        return text
    
    def _read_pdf_bytes(self, content: bytes) -> str:
        """Read PDF from bytes."""
        if not PYPDF2_AVAILABLE:
            raise ImportError("PyPDF2 is required for PDF processing")
        
        text = ""
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
        except Exception as e:
            logger.error(f"Error reading PDF bytes: {e}")
            raise
        
        return text
    
    def _read_markdown(self, path: Path) -> str:
        """Read markdown file."""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        # Optionally convert to plain text
        if MARKDOWN_AVAILABLE:
            # Simple conversion - remove HTML tags
            html = markdown.markdown(content)
            # Strip HTML tags
            text = re.sub(r'<[^>]+>', '', html)
            return text
        
        return content


def process_file(file_path: Union[str, Path], file_content: Optional[bytes] = None) -> Document:
    """Convenience function to process a file.
    
    Args:
        file_path: Path to the file.
        file_content: Optional file content bytes.
        
    Returns:
        Processed Document.
    """
    processor = DocumentProcessor()
    return processor.process(file_path, file_content)
