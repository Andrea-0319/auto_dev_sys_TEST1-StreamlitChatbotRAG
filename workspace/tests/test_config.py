"""Unit tests for config.settings module."""

import os
import pytest
from pathlib import Path

from config.settings import Config, settings


@pytest.mark.unit
class TestConfigDefaults:
    """Test default configuration values."""
    
    def test_default_model_settings(self):
        """Test default model configuration."""
        config = Config()
        
        assert config.MODEL_NAME == "Qwen/Qwen2.5-1.5B-Instruct"
        assert config.EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.DEVICE == "cpu"
        assert config.MAX_MEMORY_MB == 4000
    
    def test_default_inference_params(self):
        """Test default inference parameters."""
        config = Config()
        
        assert config.TEMPERATURE == 0.7
        assert config.MAX_TOKENS == 1024
        assert config.TOP_P == 0.9
        assert config.TOP_K == 50
    
    def test_default_rag_settings(self):
        """Test default RAG configuration."""
        config = Config()
        
        assert config.CHUNK_SIZE == 512
        assert config.CHUNK_OVERLAP == 50
        assert config.TOP_K_RETRIEVAL == 5
        assert config.MAX_CONTEXT_TOKENS == 1500
    
    def test_default_file_settings(self):
        """Test default file upload settings."""
        config = Config()
        
        assert config.MAX_FILE_SIZE_MB == 10
        assert config.ALLOWED_EXTENSIONS == (".pdf", ".txt", ".md")
    
    def test_default_ui_settings(self):
        """Test default UI settings."""
        config = Config()
        
        assert config.PAGE_TITLE == "RAG Chatbot"
        assert config.PAGE_ICON == "🤖"
        assert config.MAX_CHAT_HISTORY == 10
    
    def test_default_paths(self):
        """Test default directory paths."""
        config = Config()
        
        assert config.DATA_DIR == "data"
        assert config.DOCUMENTS_DIR == "data/documents"
        assert config.VECTOR_STORE_DIR == "data/vector_store"
        assert config.CACHE_DIR == "data/cache"


@pytest.mark.unit
class TestConfigFromEnv:
    """Test configuration from environment variables."""
    
    def test_env_model_name(self, monkeypatch):
        """Test MODEL_NAME from environment."""
        monkeypatch.setenv("MODEL_NAME", "custom-model")
        config = Config()
        assert config.MODEL_NAME == "custom-model"
    
    def test_env_embedding_model(self, monkeypatch):
        """Test EMBEDDING_MODEL from environment."""
        monkeypatch.setenv("EMBEDDING_MODEL", "custom-embeddings")
        config = Config()
        assert config.EMBEDDING_MODEL == "custom-embeddings"
    
    def test_env_temperature(self, monkeypatch):
        """Test TEMPERATURE from environment."""
        monkeypatch.setenv("TEMPERATURE", "0.5")
        config = Config()
        assert config.TEMPERATURE == 0.5
    
    def test_env_max_tokens(self, monkeypatch):
        """Test MAX_TOKENS from environment."""
        monkeypatch.setenv("MAX_TOKENS", "512")
        config = Config()
        assert config.MAX_TOKENS == 512
    
    def test_env_chunk_size(self, monkeypatch):
        """Test CHUNK_SIZE from environment."""
        monkeypatch.setenv("CHUNK_SIZE", "256")
        config = Config()
        assert config.CHUNK_SIZE == 256
    
    def test_env_max_file_size(self, monkeypatch):
        """Test MAX_FILE_SIZE_MB from environment."""
        monkeypatch.setenv("MAX_FILE_SIZE_MB", "5")
        config = Config()
        assert config.MAX_FILE_SIZE_MB == 5
    
    def test_env_device(self, monkeypatch):
        """Test DEVICE from environment."""
        monkeypatch.setenv("DEVICE", "cuda")
        config = Config()
        assert config.DEVICE == "cuda"
    
    def test_env_max_memory_none(self, monkeypatch):
        """Test MAX_MEMORY_MB as None."""
        monkeypatch.setenv("MAX_MEMORY_MB", "")
        config = Config()
        assert config.MAX_MEMORY_MB is None


@pytest.mark.unit
class TestConfigPostInit:
    """Test Config post-initialization."""
    
    def test_directories_created(self, temp_dir, monkeypatch):
        """Test that directories are created in post_init."""
        monkeypatch.setenv("DATA_DIR", str(Path(temp_dir) / "test_data"))
        monkeypatch.setenv("DOCUMENTS_DIR", str(Path(temp_dir) / "test_docs"))
        monkeypatch.setenv("VECTOR_STORE_DIR", str(Path(temp_dir) / "test_vectors"))
        monkeypatch.setenv("CACHE_DIR", str(Path(temp_dir) / "test_cache"))
        
        config = Config()
        
        assert Path(config.DATA_DIR).exists()
        assert Path(config.DOCUMENTS_DIR).exists()
        assert Path(config.VECTOR_STORE_DIR).exists()
        assert Path(config.CACHE_DIR).exists()
    
    def test_existing_directories_not_recreated(self, temp_dir, monkeypatch):
        """Test that existing directories are not affected."""
        existing_dir = Path(temp_dir) / "existing"
        existing_dir.mkdir()
        (existing_dir / "file.txt").write_text("content")
        
        monkeypatch.setenv("DATA_DIR", str(existing_dir))
        config = Config()
        
        assert (existing_dir / "file.txt").exists()


@pytest.mark.unit
class TestGlobalSettings:
    """Test global settings instance."""
    
    def test_global_settings_is_config_instance(self):
        """Test that global settings is a Config instance."""
        assert isinstance(settings, Config)
    
    def test_global_settings_has_defaults(self):
        """Test that global settings has default values."""
        assert settings.MODEL_NAME is not None
        assert settings.CHUNK_SIZE > 0
        assert settings.MAX_FILE_SIZE_MB > 0
