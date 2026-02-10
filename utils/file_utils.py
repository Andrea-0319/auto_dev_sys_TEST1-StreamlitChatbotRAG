"""File handling utilities."""

import hashlib
import shutil
from pathlib import Path
from typing import Optional, Tuple

from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


def save_uploaded_file(uploaded_file, destination_dir: Optional[str] = None) -> Tuple[bool, str]:
    """Save an uploaded file to the documents directory.
    
    Args:
        uploaded_file: Streamlit uploaded file object.
        destination_dir: Optional destination directory.
        
    Returns:
        Tuple of (success: bool, file_path or error_message: str).
    """
    try:
        dest_dir = Path(destination_dir) if destination_dir else Path(settings.DOCUMENTS_DIR)
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = dest_dir / uploaded_file.name
        
        # Check if file already exists
        if file_path.exists():
            # Append hash to filename
            file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()[:8]
            stem = file_path.stem
            suffix = file_path.suffix
            file_path = dest_dir / f"{stem}_{file_hash}{suffix}"
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"Saved uploaded file: {file_path}")
        return True, str(file_path)
        
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        return False, str(e)


def delete_file(file_path: str) -> bool:
    """Delete a file.
    
    Args:
        file_path: Path to file.
        
    Returns:
        True if deleted successfully, False otherwise.
    """
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            logger.info(f"Deleted file: {file_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to delete file {file_path}: {e}")
        return False


def get_file_info(file_path: str) -> dict:
    """Get information about a file.
    
    Args:
        file_path: Path to file.
        
    Returns:
        Dictionary with file information.
    """
    path = Path(file_path)
    if not path.exists():
        return {}
    
    stat = path.stat()
    return {
        "name": path.name,
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "extension": path.suffix.lower(),
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
    }


def clear_directory(directory: str, exclude_patterns: Optional[list] = None) -> bool:
    """Clear all files in a directory.
    
    Args:
        directory: Path to directory.
        exclude_patterns: Optional list of patterns to exclude.
        
    Returns:
        True if cleared successfully, False otherwise.
    """
    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            return True
        
        exclude_patterns = exclude_patterns or []
        
        for item in dir_path.iterdir():
            # Check if item matches any exclude pattern
            if any(pattern in item.name for pattern in exclude_patterns):
                continue
            
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        
        logger.info(f"Cleared directory: {directory}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to clear directory {directory}: {e}")
        return False


def ensure_dir(directory: str) -> Path:
    """Ensure directory exists, create if not.
    
    Args:
        directory: Path to directory.
        
    Returns:
        Path object.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path
