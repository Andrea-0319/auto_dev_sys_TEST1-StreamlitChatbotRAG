"""Configuration module for the RAG chatbot."""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Application configuration."""
    
    # Model settings
    MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    DEVICE: str = os.getenv("DEVICE", "cpu")
    MAX_MEMORY_MB: Optional[int] = int(os.getenv("MAX_MEMORY_MB", "4000")) if os.getenv("MAX_MEMORY_MB") else None
    
    # Model inference parameters
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))
    TOP_P: float = float(os.getenv("TOP_P", "0.9"))
    TOP_K: int = int(os.getenv("TOP_K", "50"))
    
    # RAG settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "1500"))
    
    # Token tracking settings
    TOKEN_WARNING_THRESHOLD: float = float(os.getenv("TOKEN_WARNING_THRESHOLD", "0.8"))
    SUMMARY_PRESERVE_MESSAGES: int = int(os.getenv("SUMMARY_PRESERVE_MESSAGES", "4"))
    MAX_BRANCHES_PER_CONVERSATION: int = int(os.getenv("MAX_BRANCHES_PER_CONVERSATION", "10"))
    
    # Tool modes
    ENABLE_TOOLS: bool = os.getenv("ENABLE_TOOLS", "true").lower() == "true"
    TOOL_MODES: Dict = field(default_factory=lambda: {
        "general": {"name": "General Chat", "system_prompt": "You are a helpful assistant."},
        "code": {"name": "Code Assistant", "system_prompt": "You are an expert programmer. Focus on code quality, best practices, and provide clear explanations."},
        "document": {"name": "Document Analyzer", "system_prompt": "You analyze documents and extract key information, summarize content, and answer questions about the text."}
    })
    
    # File upload settings
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    ALLOWED_EXTENSIONS: tuple = (".pdf", ".txt", ".md")
    
    # UI settings
    PAGE_TITLE: str = os.getenv("PAGE_TITLE", "RAG Chatbot")
    PAGE_ICON: str = os.getenv("PAGE_ICON", "🤖")
    MAX_CHAT_HISTORY: int = int(os.getenv("MAX_CHAT_HISTORY", "10"))
    
    # Paths
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    DOCUMENTS_DIR: str = os.getenv("DOCUMENTS_DIR", "data/documents")
    VECTOR_STORE_DIR: str = os.getenv("VECTOR_STORE_DIR", "data/vector_store")
    CACHE_DIR: str = os.getenv("CACHE_DIR", "data/cache")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE", "data/app.log")
    
    def __post_init__(self):
        """Ensure directories exist."""
        for path in [self.DATA_DIR, self.DOCUMENTS_DIR, self.VECTOR_STORE_DIR, self.CACHE_DIR]:
            os.makedirs(path, exist_ok=True)


# Global config instance
settings = Config()
