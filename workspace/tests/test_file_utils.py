"""Unit tests for utils.file_utils module."""

import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from utils.file_utils import (
    save_uploaded_file,
    delete_file,
    get_file_info,
    clear_directory,
    ensure_dir,
)


@pytest.mark.unit
class TestSaveUploadedFile:
    """Test file saving functionality."""
    
    def test_save_new_file(self, temp_dir):
        """Test saving a new file."""
        mock_file = MagicMock()
        mock_file.name = "test.txt"
        mock_file.getbuffer.return_value = b"Test content"
        
        success, result = save_uploaded_file(mock_file, temp_dir)
        
        assert success is True
        assert os.path.exists(result)
        with open(result, 'rb') as f:
            assert f.read() == b"Test content"
    
    def test_save_duplicate_file(self, temp_dir):
        """Test saving a file with duplicate name."""
        mock_file = MagicMock()
        mock_file.name = "test.txt"
        mock_file.getbuffer.return_value = b"Test content"
        mock_file.getvalue.return_value = b"Test content"
        
        # Save first file
        success1, result1 = save_uploaded_file(mock_file, temp_dir)
        assert success1 is True
        
        # Save second file with same name
        success2, result2 = save_uploaded_file(mock_file, temp_dir)
        assert success2 is True
        assert result1 != result2  # Should have different names
        assert os.path.exists(result2)
    
    def test_save_without_destination(self, temp_dir, monkeypatch):
        """Test saving without specifying destination."""
        monkeypatch.setattr('utils.file_utils.settings.DOCUMENTS_DIR', temp_dir)
        
        mock_file = MagicMock()
        mock_file.name = "document.pdf"
        mock_file.getbuffer.return_value = b"PDF content"
        
        success, result = save_uploaded_file(mock_file)
        
        assert success is True
        assert temp_dir in result
        assert os.path.exists(result)
    
    def test_save_failure(self, temp_dir):
        """Test handling save failures."""
        mock_file = MagicMock()
        mock_file.name = "test.txt"
        mock_file.getbuffer.side_effect = Exception("Write error")
        
        success, error_msg = save_uploaded_file(mock_file, temp_dir)
        
        assert success is False
        assert "Write error" in error_msg


@pytest.mark.unit
class TestDeleteFile:
    """Test file deletion functionality."""
    
    def test_delete_existing_file(self, temp_dir):
        """Test deleting an existing file."""
        test_file = Path(temp_dir) / "to_delete.txt"
        test_file.write_text("Delete me")
        
        assert test_file.exists()
        result = delete_file(str(test_file))
        
        assert result is True
        assert not test_file.exists()
    
    def test_delete_nonexistent_file(self, temp_dir):
        """Test deleting a non-existent file."""
        nonexistent = str(Path(temp_dir) / "does_not_exist.txt")
        
        result = delete_file(nonexistent)
        
        assert result is False
    
    def test_delete_failure(self, temp_dir):
        """Test handling delete failures."""
        test_file = Path(temp_dir) / "protected.txt"
        test_file.write_text("Content")
        
        # Make file read-only
        os.chmod(str(test_file), 0o444)
        
        try:
            result = delete_file(str(test_file))
            # Result may vary by OS, but should not crash
            assert isinstance(result, bool)
        finally:
            # Restore permissions for cleanup
            os.chmod(str(test_file), 0o666)


@pytest.mark.unit
class TestGetFileInfo:
    """Test file information retrieval."""
    
    def test_get_info_existing_file(self, temp_dir):
        """Test getting info for existing file."""
        test_file = Path(temp_dir) / "info_test.txt"
        test_file.write_text("Test content for info")
        
        info = get_file_info(str(test_file))
        
        assert info["name"] == "info_test.txt"
        assert info["extension"] == ".txt"
        assert info["size_bytes"] > 0
        assert "size_mb" in info
        assert "created" in info
        assert "modified" in info
    
    def test_get_info_nonexistent_file(self):
        """Test getting info for non-existent file."""
        info = get_file_info("/path/to/nonexistent/file.txt")
        
        assert info == {}
    
    def test_get_info_different_extensions(self, temp_dir):
        """Test getting info for different file types."""
        # PDF file
        pdf_file = Path(temp_dir) / "doc.PDF"
        pdf_file.write_text("PDF content")
        info = get_file_info(str(pdf_file))
        assert info["extension"] == ".pdf"
        
        # Markdown file
        md_file = Path(temp_dir) / "notes.MD"
        md_file.write_text("Markdown content")
        info = get_file_info(str(md_file))
        assert info["extension"] == ".md"


@pytest.mark.unit
class TestClearDirectory:
    """Test directory clearing functionality."""
    
    def test_clear_all_files(self, temp_dir):
        """Test clearing all files from directory."""
        # Create test files
        (Path(temp_dir) / "file1.txt").write_text("Content 1")
        (Path(temp_dir) / "file2.txt").write_text("Content 2")
        subdir = Path(temp_dir) / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("Content 3")
        
        result = clear_directory(temp_dir)
        
        assert result is True
        assert len(list(Path(temp_dir).iterdir())) == 0
    
    def test_clear_with_exclusions(self, temp_dir):
        """Test clearing with excluded patterns."""
        # Create test files
        (Path(temp_dir) / "keep.txt").write_text("Keep")
        (Path(temp_dir) / "delete.txt").write_text("Delete")
        (Path(temp_dir) / "data.json").write_text("JSON")
        
        result = clear_directory(temp_dir, exclude_patterns=["keep", ".json"])
        
        assert result is True
        assert (Path(temp_dir) / "keep.txt").exists()
        assert (Path(temp_dir) / "data.json").exists()
        assert not (Path(temp_dir) / "delete.txt").exists()
    
    def test_clear_nonexistent_directory(self):
        """Test clearing non-existent directory."""
        result = clear_directory("/path/that/does/not/exist")
        
        assert result is True  # Returns True if directory doesn't exist
    
    def test_clear_failure(self, temp_dir):
        """Test handling clear failures."""
        # Create a file with no write permissions
        protected_file = Path(temp_dir) / "protected.txt"
        protected_file.write_text("Protected")
        os.chmod(str(protected_file), 0o444)
        
        # Try to clear directory containing protected file
        # This should handle the error gracefully
        result = clear_directory(temp_dir)
        
        # Restore permissions
        os.chmod(str(protected_file), 0o666)
        
        # Result depends on OS, but shouldn't crash
        assert isinstance(result, bool)


@pytest.mark.unit
class TestEnsureDir:
    """Test directory creation functionality."""
    
    def test_create_new_directory(self, temp_dir):
        """Test creating a new directory."""
        new_dir = Path(temp_dir) / "new" / "nested" / "dir"
        
        assert not new_dir.exists()
        result = ensure_dir(str(new_dir))
        
        assert new_dir.exists()
        assert isinstance(result, Path)
        assert str(result) == str(new_dir)
    
    def test_ensure_existing_directory(self, temp_dir):
        """Test ensuring an existing directory."""
        existing_dir = Path(temp_dir) / "existing"
        existing_dir.mkdir()
        (existing_dir / "file.txt").write_text("Content")
        
        result = ensure_dir(str(existing_dir))
        
        assert existing_dir.exists()
        assert (existing_dir / "file.txt").exists()
        assert isinstance(result, Path)
    
    def test_return_type(self, temp_dir):
        """Test that ensure_dir returns a Path object."""
        result = ensure_dir(temp_dir)
        
        assert isinstance(result, Path)
