"""Validation utilities."""

import re
from pathlib import Path
from typing import Tuple

from config import settings


def validate_file_extension(filename: str) -> bool:
    """Check if file has allowed extension.
    
    Args:
        filename: Name of the file.
        
    Returns:
        True if extension is allowed, False otherwise.
    """
    return any(filename.lower().endswith(ext) for ext in settings.ALLOWED_EXTENSIONS)


def validate_file_size(file_size_bytes: int) -> bool:
    """Check if file size is within limits.
    
    Args:
        file_size_bytes: Size of file in bytes.
        
    Returns:
        True if size is valid, False otherwise.
    """
    max_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    return file_size_bytes <= max_size_bytes


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent security issues.
    
    Args:
        filename: Original filename.
        
    Returns:
        Sanitized filename.
    """
    # Remove path separators and null bytes
    filename = re.sub(r'[\\/\\x00]', '', filename)
    # Remove leading dots
    filename = filename.lstrip('.')
    # Limit length
    if len(filename) > 255:
        name, ext = Path(filename).stem, Path(filename).suffix
        filename = name[:255-len(ext)] + ext
    return filename


def validate_user_input(text: str) -> Tuple[bool, str]:
    """Validate user input for safety.
    
    Args:
        text: User input text.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    if not text or not text.strip():
        return False, "Input cannot be empty."
    
    if len(text) > 10000:
        return False, "Input is too long (max 10,000 characters)."
    
    # Check for potential prompt injection patterns
    suspicious_patterns = [
        r"ignore previous instructions",
        r"disregard.*prompt",
        r"system prompt",
        r"you are now.*assistant",
    ]
    
    text_lower = text.lower()
    for pattern in suspicious_patterns:
        if re.search(pattern, text_lower):
            return False, "Input contains potentially harmful content."
    
    return True, ""


def escape_special_chars(text: str) -> str:
    """Escape special characters in text.
    
    Args:
        text: Input text.
        
    Returns:
        Escaped text.
    """
    # Escape HTML special characters
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text
