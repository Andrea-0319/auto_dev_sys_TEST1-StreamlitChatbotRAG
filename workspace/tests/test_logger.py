"""Unit tests for utils.logger module."""

import os
import logging
import pytest
from pathlib import Path

from utils.logger import setup_logger


@pytest.mark.unit
class TestSetupLogger:
    """Test logger setup functionality."""
    
    def test_logger_creation(self, temp_dir, monkeypatch):
        """Test basic logger creation."""
        monkeypatch.setattr('utils.logger.settings.LOG_FILE', None)
        monkeypatch.setattr('utils.logger.settings.LOG_LEVEL', 'INFO')
        
        logger = setup_logger("test_logger_1")
        
        assert logger.name == "test_logger_1"
        assert logger.level == logging.INFO
    
    def test_logger_with_file_handler(self, temp_dir, monkeypatch):
        """Test logger with file handler."""
        log_file = str(Path(temp_dir) / "test.log")
        monkeypatch.setattr('utils.logger.settings.LOG_FILE', log_file)
        monkeypatch.setattr('utils.logger.settings.LOG_LEVEL', 'DEBUG')
        
        logger = setup_logger("test_logger_2")
        
        # Check that file handler was added
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
        assert len(file_handlers) > 0
    
    def test_logger_prevent_duplicate_handlers(self, temp_dir, monkeypatch):
        """Test that duplicate handlers are not added."""
        monkeypatch.setattr('utils.logger.settings.LOG_FILE', None)
        
        logger1 = setup_logger("test_logger_3")
        initial_handler_count = len(logger1.handlers)
        
        # Call setup_logger again with same name
        logger2 = setup_logger("test_logger_3")
        
        assert logger1 is logger2
        assert len(logger2.handlers) == initial_handler_count
    
    def test_log_levels(self, temp_dir, monkeypatch):
        """Test different log levels."""
        monkeypatch.setattr('utils.logger.settings.LOG_FILE', None)
        
        # Test DEBUG level
        monkeypatch.setattr('utils.logger.settings.LOG_LEVEL', 'DEBUG')
        logger_debug = setup_logger("test_debug")
        assert logger_debug.level == logging.DEBUG
        
        # Test ERROR level
        monkeypatch.setattr('utils.logger.settings.LOG_LEVEL', 'ERROR')
        logger_error = setup_logger("test_error")
        assert logger_error.level == logging.ERROR
    
    def test_log_file_creation(self, temp_dir, monkeypatch):
        """Test that log file is created."""
        log_file = str(Path(temp_dir) / "app.log")
        monkeypatch.setattr('utils.logger.settings.LOG_FILE', log_file)
        monkeypatch.setattr('utils.logger.settings.LOG_LEVEL', 'INFO')
        
        logger = setup_logger("test_file_logger")
        logger.info("Test message")
        
        # Check that log file was created
        assert Path(log_file).exists()
        
        # Check content
        content = Path(log_file).read_text()
        assert "Test message" in content


@pytest.mark.unit
class TestLoggerFormatting:
    """Test logger formatting."""
    
    def test_console_format(self, temp_dir, monkeypatch):
        """Test console handler format."""
        monkeypatch.setattr('utils.logger.settings.LOG_FILE', None)
        monkeypatch.setattr('utils.logger.settings.LOG_LEVEL', 'INFO')
        
        logger = setup_logger("test_format")
        
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.handlers.RotatingFileHandler)]
        if console_handlers:
            formatter = console_handlers[0].formatter
            format_str = formatter._fmt
            
            # Check format includes expected fields
            assert "%(asctime)s" in format_str
            assert "%(name)s" in format_str
            assert "%(levelname)s" in format_str
            assert "%(message)s" in format_str
    
    def test_file_format(self, temp_dir, monkeypatch):
        """Test file handler format includes more detail."""
        log_file = str(Path(temp_dir) / "format_test.log")
        monkeypatch.setattr('utils.logger.settings.LOG_FILE', log_file)
        monkeypatch.setattr('utils.logger.settings.LOG_LEVEL', 'DEBUG')
        
        logger = setup_logger("test_file_format")
        
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
        if file_handlers:
            formatter = file_handlers[0].formatter
            format_str = formatter._fmt
            
            # File format should include function name and line number
            assert "%(funcName)s" in format_str
            assert "%(lineno)d" in format_str
