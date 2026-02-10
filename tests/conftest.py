"""Test configuration and fixtures."""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Add workspace to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_text():
    """Provide sample text for testing."""
    return """
    This is a sample document for testing purposes.
    
    It contains multiple paragraphs to test chunking functionality.
    The document discusses various topics including artificial intelligence,
    machine learning, and natural language processing.
    
    AI has transformed many industries. Machine learning algorithms can
    now recognize patterns in data that humans might miss. Natural language
    processing enables computers to understand and generate human language.
    
    These technologies continue to evolve rapidly.
    """


@pytest.fixture
def sample_chunks():
    """Provide sample chunk dictionaries."""
    return [
        {
            "id": "chunk-1",
            "text": "This is the first chunk of text.",
            "document_id": "doc-1",
            "document_name": "test.txt",
            "page_number": None,
        },
        {
            "id": "chunk-2",
            "text": "This is the second chunk with different content.",
            "document_id": "doc-1",
            "document_name": "test.txt",
            "page_number": None,
        },
        {
            "id": "chunk-3",
            "text": "Third chunk contains more information about testing.",
            "document_id": "doc-2",
            "document_name": "test2.txt",
            "page_number": None,
        },
    ]


@pytest.fixture
def mock_uploaded_file(temp_dir):
    """Create a mock uploaded file object."""
    mock_file = MagicMock()
    mock_file.name = "test.txt"
    mock_file.getvalue.return_value = b"Test file content"
    mock_file.getbuffer.return_value = b"Test file content"
    return mock_file


@pytest.fixture
def mock_session_state():
    """Mock Streamlit session state."""
    with patch('streamlit.session_state', {}) as mock_state:
        yield mock_state


@pytest.fixture
def clean_config():
    """Provide a clean config instance for testing."""
    from config.settings import Config
    config = Config()
    yield config


@pytest.fixture
def mock_embeddings():
    """Provide mock embedding vectors."""
    import numpy as np
    return np.random.randn(10, 384).astype(np.float32)


@pytest.fixture(scope="function")
def isolated_vector_store():
    """Create an isolated vector store for testing."""
    from rag.vector_store import VectorStore
    store = VectorStore(dimension=384)
    yield store
    store.clear()


@pytest.fixture
def sample_document():
    """Provide a sample document."""
    from datetime import datetime
    from rag.document_processor import Document
    return Document(
        id="test-doc-1",
        name="test.txt",
        type="txt",
        content="This is test content for the document processor.",
        chunks=[],
        upload_time=datetime.now(),
        size_bytes=100,
    )


@pytest.fixture
def mock_model_response():
    """Provide a mock model response with reasoning."""
    return """REASONING:
    Step 1: I need to analyze the user's query about artificial intelligence.
    Step 2: Looking at the context, I see information about machine learning.
    Step 3: The context mentions NLP and pattern recognition.
    
    ANSWER:
    Based on the context provided, artificial intelligence encompasses machine learning 
    and natural language processing. These technologies enable pattern recognition 
    and language understanding."""


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances before each test."""
    # Reset embedding service
    import rag.embeddings as emb_module
    emb_module._embedding_service = None
    
    # Reset session manager
    import core.session as sess_module
    sess_module._session_manager = None
    
    yield


@pytest.fixture
def test_env_vars():
    """Set test environment variables."""
    env_vars = {
        "MODEL_NAME": "test-model",
        "EMBEDDING_MODEL": "test-embedding-model",
        "DEVICE": "cpu",
        "MAX_MEMORY_MB": "1000",
        "TEMPERATURE": "0.5",
        "MAX_TOKENS": "512",
        "CHUNK_SIZE": "256",
        "CHUNK_OVERLAP": "25",
        "TOP_K_RETRIEVAL": "3",
        "MAX_FILE_SIZE_MB": "5",
        "LOG_LEVEL": "DEBUG",
    }
    
    original_values = {}
    for key, value in env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield env_vars
    
    for key, value in original_values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# Markers for different test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.functional = pytest.mark.functional
pytest.mark.edge_case = pytest.mark.edge_case
pytest.mark.error_handling = pytest.mark.error_handling
