"""Token streaming handler."""

import threading
import time
from typing import Generator, Callable, Optional
from dataclasses import dataclass

from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class StreamState:
    """State of the streaming process."""
    is_streaming: bool = False
    is_stopped: bool = False
    tokens_generated: int = 0
    start_time: Optional[float] = None
    
    def reset(self):
        """Reset state."""
        self.is_streaming = False
        self.is_stopped = False
        self.tokens_generated = 0
        self.start_time = None


class StreamHandler:
    """Handle token streaming with control mechanisms."""
    
    def __init__(self):
        self.state = StreamState()
        self._lock = threading.Lock()
    
    def start(self):
        """Start streaming."""
        with self._lock:
            self.state.reset()
            self.state.is_streaming = True
            self.state.start_time = time.time()
        logger.debug("Stream started")
    
    def stop(self):
        """Stop streaming."""
        with self._lock:
            self.state.is_stopped = True
            self.state.is_streaming = False
        logger.debug("Stream stopped")
    
    def pause(self):
        """Pause streaming."""
        with self._lock:
            self.state.is_streaming = False
    
    def resume(self):
        """Resume streaming."""
        with self._lock:
            if not self.state.is_stopped:
                self.state.is_streaming = True
    
    def add_token(self):
        """Increment token counter."""
        with self._lock:
            self.state.tokens_generated += 1
    
    @property
    def is_active(self) -> bool:
        """Check if streaming is active."""
        with self._lock:
            return self.state.is_streaming and not self.state.is_stopped
    
    @property
    def should_stop(self) -> bool:
        """Check if streaming should stop."""
        with self._lock:
            return self.state.is_stopped
    
    def get_stats(self) -> dict:
        """Get streaming statistics."""
        with self._lock:
            elapsed = time.time() - self.state.start_time if self.state.start_time else 0
            tokens_per_sec = self.state.tokens_generated / elapsed if elapsed > 0 else 0
            
            return {
                "tokens_generated": self.state.tokens_generated,
                "elapsed_seconds": elapsed,
                "tokens_per_second": tokens_per_sec,
                "is_streaming": self.state.is_streaming,
                "is_stopped": self.state.is_stopped,
            }


class TokenBuffer:
    """Buffer for token streaming with rate limiting."""
    
    def __init__(self, max_tokens: Optional[int] = None, min_delay: float = 0.0):
        """Initialize token buffer.
        
        Args:
            max_tokens: Maximum tokens to generate.
            min_delay: Minimum delay between tokens in seconds.
        """
        self.max_tokens = max_tokens
        self.min_delay = min_delay
        self._buffer = []
        self._token_count = 0
        self._last_time = 0
    
    def add(self, token: str) -> bool:
        """Add token to buffer.
        
        Args:
            token: Token string.
            
        Returns:
            True if token was added, False if limit reached.
        """
        if self.max_tokens and self._token_count >= self.max_tokens:
            return False
        
        # Apply rate limiting
        if self.min_delay > 0:
            current_time = time.time()
            elapsed = current_time - self._last_time
            if elapsed < self.min_delay:
                time.sleep(self.min_delay - elapsed)
            self._last_time = time.time()
        
        self._buffer.append(token)
        self._token_count += 1
        return True
    
    def get_all(self) -> str:
        """Get all buffered tokens as string."""
        return "".join(self._buffer)
    
    def clear(self):
        """Clear buffer."""
        self._buffer = []
        self._token_count = 0
    
    @property
    def count(self) -> int:
        """Get token count."""
        return self._token_count


def create_streaming_generator(
    generator: Generator[str, None, None],
    stream_handler: StreamHandler,
    token_buffer: Optional[TokenBuffer] = None
) -> Generator[str, None, None]:
    """Wrap a generator with streaming controls.
    
    Args:
        generator: Base token generator.
        stream_handler: Stream handler for control.
        token_buffer: Optional token buffer.
        
    Yields:
        Tokens from generator.
    """
    stream_handler.start()
    
    try:
        for token in generator:
            if stream_handler.should_stop:
                logger.debug("Stream stopped by user")
                break
            
            stream_handler.add_token()
            
            if token_buffer:
                if not token_buffer.add(token):
                    logger.debug("Token buffer limit reached")
                    break
            
            yield token
            
    except Exception as e:
        logger.error(f"Error in streaming: {e}")
        raise
    finally:
        stream_handler.stop()
