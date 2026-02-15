"""Token tracking service for real-time token count and context usage monitoring."""

from dataclasses import dataclass
from typing import List, Optional

from transformers import AutoTokenizer

from config import settings
from core.session import Message
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class TokenBreakdown:
    """Token usage breakdown by category."""
    system_prompt_tokens: int
    chat_history_tokens: int
    context_tokens: int
    total: int
    percentage: float


@dataclass
class TokenUsage:
    """Current token usage information."""
    current: int
    max: int
    percentage: float
    breakdown: TokenBreakdown


class TokenTracker:
    """Track and calculate token usage for conversation context."""
    
    def __init__(
        self,
        model_name: str = None,
        max_context_tokens: int = None
    ):
        """Initialize token tracker.
        
        Args:
            model_name: Name of the model for tokenizer.
            max_context_tokens: Maximum context window size.
        """
        self._model_name = model_name or settings.MODEL_NAME
        self._max_context_tokens = max_context_tokens or settings.MAX_CONTEXT_TOKENS
        self._tokenizer: Optional[AutoTokenizer] = None
        self._current_usage: Optional[TokenUsage] = None
        
        self._load_tokenizer()
    
    def _load_tokenizer(self) -> None:
        """Load the tokenizer for token counting."""
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                trust_remote_code=True
            )
            logger.info(f"Loaded tokenizer for: {self._model_name}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}. Using fallback.")
            self._tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count tokens for.
            
        Returns:
            Token count.
        """
        if not text:
            return 0
        
        if self._tokenizer:
            try:
                tokens = self._tokenizer.encode(text, add_special_tokens=True)
                return len(tokens)
            except Exception as e:
                logger.warning(f"Tokenization error: {e}")
        
        return self._fallback_count_tokens(text)
    
    def _fallback_count_tokens(self, text: str) -> int:
        """Fallback token counting using character approximation.
        
        Args:
            text: Text to count.
            
        Returns:
            Approximate token count.
        """
        return len(text) // 4 + 1
    
    def get_context_breakdown(
        self,
        messages: List[Message],
        system_prompt: str,
        context_docs: List[str]
    ) -> TokenBreakdown:
        """Get token breakdown by category.
        
        Args:
            messages: List of conversation messages.
            system_prompt: System prompt text.
            context_docs: Retrieved context documents.
            
        Returns:
            Token breakdown.
        """
        system_tokens = self.count_tokens(system_prompt)
        
        chat_tokens = 0
        for msg in messages:
            chat_tokens += self.count_tokens(msg.content)
            if msg.reasoning:
                chat_tokens += self.count_tokens(msg.reasoning)
        
        context_tokens = 0
        for doc in context_docs:
            context_tokens += self.count_tokens(doc)
        
        total = system_tokens + chat_tokens + context_tokens
        percentage = (total / self._max_context_tokens) * 100 if self._max_context_tokens > 0 else 0
        
        return TokenBreakdown(
            system_prompt_tokens=system_tokens,
            chat_history_tokens=chat_tokens,
            context_tokens=context_tokens,
            total=total,
            percentage=percentage
        )
    
    def get_current_usage(
        self,
        messages: List[Message] = None,
        system_prompt: str = "",
        context_docs: List[str] = None
    ) -> TokenUsage:
        """Get current token usage.
        
        Args:
            messages: Current messages.
            system_prompt: System prompt.
            context_docs: Retrieved context.
            
        Returns:
            Current token usage.
        """
        if messages is None:
            messages = []
        if context_docs is None:
            context_docs = []
        
        breakdown = self.get_context_breakdown(messages, system_prompt, context_docs)
        
        self._current_usage = TokenUsage(
            current=breakdown.total,
            max=self._max_context_tokens,
            percentage=breakdown.percentage,
            breakdown=breakdown
        )
        
        return self._current_usage
    
    def is_approaching_limit(self, threshold: float = 0.8) -> bool:
        """Check if approaching context limit.
        
        Args:
            threshold: Percentage threshold (0-1).
            
        Returns:
            True if approaching limit.
        """
        if self._current_usage is None:
            return False
        
        return self._current_usage.percentage >= (threshold * 100)
    
    def estimate_response_tokens(self, prompt: str, max_new_tokens: int = None) -> int:
        """Estimate tokens available for response.
        
        Args:
            prompt: Input prompt.
            max_new_tokens: Desired max new tokens.
            
        Returns:
            Available tokens for response.
        """
        prompt_tokens = self.count_tokens(prompt)
        available = self._max_context_tokens - prompt_tokens
        
        if max_new_tokens:
            return min(available, max_new_tokens)
        
        return max(0, available)
    
    def get_token_warning_level(self) -> str:
        """Get warning level based on token usage.
        
        Returns:
            Warning level: "normal", "warning", or "critical".
        """
        if self._current_usage is None:
            return "normal"
        
        pct = self._current_usage.percentage
        
        if pct >= 90:
            return "critical"
        elif pct >= 80:
            return "warning"
        return "normal"


_token_tracker: Optional[TokenTracker] = None


def get_token_tracker() -> TokenTracker:
    """Get or create global token tracker."""
    global _token_tracker
    if _token_tracker is None:
        _token_tracker = TokenTracker()
    return _token_tracker
