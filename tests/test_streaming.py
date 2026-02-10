"""Unit tests for model.streaming module."""

import time
import pytest
from unittest.mock import Mock

from model.streaming import StreamState, StreamHandler, TokenBuffer, create_streaming_generator


@pytest.mark.unit
class TestStreamState:
    """Test StreamState dataclass."""
    
    def test_default_initialization(self):
        """Test StreamState default initialization."""
        state = StreamState()
        
        assert state.is_streaming is False
        assert state.is_stopped is False
        assert state.tokens_generated == 0
        assert state.start_time is None
    
    def test_reset(self):
        """Test StreamState reset."""
        state = StreamState()
        state.is_streaming = True
        state.is_stopped = True
        state.tokens_generated = 10
        state.start_time = time.time()
        
        state.reset()
        
        assert state.is_streaming is False
        assert state.is_stopped is False
        assert state.tokens_generated == 0
        assert state.start_time is None


@pytest.mark.unit
class TestStreamHandler:
    """Test StreamHandler class."""
    
    def test_initialization(self):
        """Test StreamHandler initialization."""
        handler = StreamHandler()
        
        assert isinstance(handler.state, StreamState)
        assert handler.is_active is False
        assert handler.should_stop is False
    
    def test_start(self):
        """Test starting stream."""
        handler = StreamHandler()
        handler.start()
        
        assert handler.state.is_streaming is True
        assert handler.state.is_stopped is False
        assert handler.state.tokens_generated == 0
        assert handler.state.start_time is not None
    
    def test_stop(self):
        """Test stopping stream."""
        handler = StreamHandler()
        handler.start()
        handler.stop()
        
        assert handler.state.is_stopped is True
        assert handler.state.is_streaming is False
        assert handler.should_stop is True
    
    def test_pause_resume(self):
        """Test pause and resume."""
        handler = StreamHandler()
        handler.start()
        
        handler.pause()
        assert handler.is_active is False
        
        handler.resume()
        assert handler.is_active is True
    
    def test_resume_after_stop(self):
        """Test resume after stop doesn't restart."""
        handler = StreamHandler()
        handler.start()
        handler.stop()
        
        handler.resume()
        assert handler.is_active is False  # Should still be stopped
    
    def test_add_token(self):
        """Test adding tokens."""
        handler = StreamHandler()
        handler.start()
        
        handler.add_token()
        handler.add_token()
        handler.add_token()
        
        assert handler.state.tokens_generated == 3
    
    def test_get_stats(self):
        """Test getting stream statistics."""
        handler = StreamHandler()
        handler.start()
        handler.add_token()
        handler.add_token()
        
        stats = handler.get_stats()
        
        assert stats["tokens_generated"] == 2
        assert stats["is_streaming"] is True
        assert stats["is_stopped"] is False
        assert stats["elapsed_seconds"] >= 0
        assert stats["tokens_per_second"] >= 0
    
    def test_thread_safety(self):
        """Test thread-safe operations."""
        import threading
        
        handler = StreamHandler()
        handler.start()
        
        def add_tokens():
            for _ in range(100):
                handler.add_token()
        
        threads = [threading.Thread(target=add_tokens) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert handler.state.tokens_generated == 500


@pytest.mark.unit
class TestTokenBuffer:
    """Test TokenBuffer class."""
    
    def test_initialization_defaults(self):
        """Test TokenBuffer default initialization."""
        buffer = TokenBuffer()
        
        assert buffer.max_tokens is None
        assert buffer.min_delay == 0.0
        assert buffer.count == 0
    
    def test_initialization_custom(self):
        """Test TokenBuffer custom initialization."""
        buffer = TokenBuffer(max_tokens=100, min_delay=0.01)
        
        assert buffer.max_tokens == 100
        assert buffer.min_delay == 0.01
    
    def test_add_token(self):
        """Test adding tokens."""
        buffer = TokenBuffer()
        
        result1 = buffer.add("Hello")
        result2 = buffer.add(" ")
        result3 = buffer.add("World")
        
        assert result1 is True
        assert result2 is True
        assert result3 is True
        assert buffer.count == 3
    
    def test_max_tokens_limit(self):
        """Test max tokens limit."""
        buffer = TokenBuffer(max_tokens=3)
        
        buffer.add("A")
        buffer.add("B")
        buffer.add("C")
        result = buffer.add("D")  # Should fail
        
        assert result is False
        assert buffer.count == 3
    
    def test_rate_limiting(self):
        """Test rate limiting."""
        buffer = TokenBuffer(min_delay=0.05)  # 50ms delay
        
        start = time.time()
        buffer.add("A")
        buffer.add("B")
        buffer.add("C")
        elapsed = time.time() - start
        
        # Should take at least 100ms (2 delays)
        assert elapsed >= 0.1
    
    def test_get_all(self):
        """Test getting all tokens."""
        buffer = TokenBuffer()
        buffer.add("Hello")
        buffer.add(" ")
        buffer.add("World")
        
        result = buffer.get_all()
        
        assert result == "Hello World"
    
    def test_clear(self):
        """Test clearing buffer."""
        buffer = TokenBuffer()
        buffer.add("Test")
        buffer.add("Content")
        
        buffer.clear()
        
        assert buffer.count == 0
        assert buffer.get_all() == ""


@pytest.mark.unit
class TestCreateStreamingGenerator:
    """Test create_streaming_generator function."""
    
    def test_streaming_generator(self):
        """Test streaming generator."""
        def mock_generator():
            yield "Hello"
            yield " "
            yield "World"
        
        handler = StreamHandler()
        wrapped = create_streaming_generator(mock_generator(), handler)
        
        tokens = list(wrapped)
        
        assert tokens == ["Hello", " ", "World"]
        assert handler.state.tokens_generated == 3
        assert handler.state.is_stopped is True  # Stopped after completion
    
    def test_streaming_generator_stop(self):
        """Test stopping streaming generator."""
        def mock_generator():
            for i in range(100):
                yield f"token{i}"
        
        handler = StreamHandler()
        wrapped = create_streaming_generator(mock_generator(), handler)
        
        tokens = []
        for token in wrapped:
            tokens.append(token)
            if len(tokens) == 5:
                handler.stop()
        
        assert len(tokens) == 5
        assert handler.state.is_stopped is True
    
    def test_streaming_generator_with_buffer(self):
        """Test streaming generator with token buffer."""
        def mock_generator():
            yield "A"
            yield "B"
            yield "C"
            yield "D"
        
        handler = StreamHandler()
        buffer = TokenBuffer(max_tokens=3)
        
        wrapped = create_streaming_generator(mock_generator(), handler, buffer)
        tokens = list(wrapped)
        
        assert len(tokens) == 3  # Limited by buffer
        assert buffer.get_all() == "ABC"
    
    def test_streaming_generator_error(self):
        """Test streaming generator error handling."""
        def error_generator():
            yield "A"
            raise ValueError("Test error")
        
        handler = StreamHandler()
        wrapped = create_streaming_generator(error_generator(), handler)
        
        with pytest.raises(ValueError):
            list(wrapped)
        
        assert handler.state.is_stopped is True
