"""Unit tests for core.token_tracker module."""

import pytest
from unittest.mock import MagicMock, patch

from core.token_tracker import TokenTracker, TokenBreakdown, TokenUsage, get_token_tracker
from core.session import Message


@pytest.mark.unit
class TestTokenBreakdown:
    """Test TokenBreakdown dataclass."""
    
    def test_token_breakdown_creation(self):
        """Test TokenBreakdown creation."""
        breakdown = TokenBreakdown(
            system_prompt_tokens=100,
            chat_history_tokens=200,
            context_tokens=50,
            total=350,
            percentage=23.33
        )
        
        assert breakdown.system_prompt_tokens == 100
        assert breakdown.chat_history_tokens == 200
        assert breakdown.context_tokens == 50
        assert breakdown.total == 350
        assert breakdown.percentage == 23.33


@pytest.mark.unit
class TestTokenUsage:
    """Test TokenUsage dataclass."""
    
    def test_token_usage_creation(self):
        """Test TokenUsage creation."""
        breakdown = TokenBreakdown(
            system_prompt_tokens=100,
            chat_history_tokens=200,
            context_tokens=50,
            total=350,
            percentage=23.33
        )
        
        usage = TokenUsage(
            current=350,
            max=1500,
            percentage=23.33,
            breakdown=breakdown
        )
        
        assert usage.current == 350
        assert usage.max == 1500
        assert usage.percentage == 23.33
        assert usage.breakdown == breakdown


@pytest.mark.unit
class TestTokenTrackerInitialization:
    """Test TokenTracker initialization."""
    
    def test_initialization_default(self):
        """Test TokenTracker default initialization."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            mock_tokenizer_instance.encode.return_value = [1, 2, 3]
            
            tracker = TokenTracker()
            
            assert tracker._max_context_tokens == 1500
            assert tracker._model_name == "Qwen/Qwen2.5-1.5B-Instruct"
    
    def test_initialization_custom(self):
        """Test TokenTracker with custom parameters."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            tracker = TokenTracker(
                model_name="test-model",
                max_context_tokens=2000
            )
            
            assert tracker._max_context_tokens == 2000
            assert tracker._model_name == "test-model"


@pytest.mark.unit
class TestTokenTrackerCountTokens:
    """Test token counting functionality."""
    
    def test_count_tokens_empty_string(self):
        """Test counting tokens for empty string."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            tracker = TokenTracker()
            result = tracker.count_tokens("")
            
            assert result == 0
    
    def test_count_tokens_with_tokenizer(self):
        """Test counting tokens using tokenizer."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
            
            tracker = TokenTracker()
            result = tracker.count_tokens("Hello world")
            
            assert result == 5
            mock_tokenizer_instance.encode.assert_called_once()
    
    def test_count_tokens_fallback(self):
        """Test fallback token counting when tokenizer fails."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            mock_tokenizer_instance.encode.side_effect = Exception("Tokenizer error")
            
            tracker = TokenTracker()
            tracker._tokenizer = mock_tokenizer_instance
            
            result = tracker.count_tokens("Hello world")
            
            assert result == len("Hello world") // 4 + 1
    
    def test_count_tokens_no_tokenizer(self):
        """Test fallback token counting when no tokenizer."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer.from_pretrained.side_effect = Exception("Load error")
            
            tracker = TokenTracker()
            
            result = tracker.count_tokens("Hello world test")
            
            assert result == len("Hello world test") // 4 + 1


@pytest.mark.unit
class TestTokenTrackerContextBreakdown:
    """Test context breakdown functionality."""
    
    def test_get_context_breakdown_empty(self):
        """Test context breakdown with empty inputs."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            mock_tokenizer_instance.encode.return_value = [1]
            
            tracker = TokenTracker()
            
            breakdown = tracker.get_context_breakdown(
                messages=[],
                system_prompt="",
                context_docs=[]
            )
            
            assert breakdown.system_prompt_tokens == 0
            assert breakdown.chat_history_tokens == 0
            assert breakdown.context_tokens == 0
            assert breakdown.total == 0
    
    def test_get_context_breakdown_with_messages(self):
        """Test context breakdown with messages."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            mock_tokenizer_instance.encode.return_value = [1, 2, 3]
            
            tracker = TokenTracker()
            
            messages = [
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there")
            ]
            
            breakdown = tracker.get_context_breakdown(
                messages=messages,
                system_prompt="You are a helpful assistant.",
                context_docs=["Context doc 1", "Context doc 2"]
            )
            
            assert breakdown.system_prompt_tokens > 0
            assert breakdown.chat_history_tokens > 0
            assert breakdown.context_tokens > 0
    
    def test_get_context_breakdown_with_reasoning(self):
        """Test context breakdown includes reasoning tokens."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            mock_tokenizer_instance.encode.return_value = [1, 2]
            
            tracker = TokenTracker()
            
            messages = [
                Message(
                    role="assistant",
                    content="Answer",
                    reasoning="Thinking process"
                )
            ]
            
            breakdown = tracker.get_context_breakdown(
                messages=messages,
                system_prompt="System",
                context_docs=[]
            )
            
            assert breakdown.chat_history_tokens > 0


@pytest.mark.unit
class TestTokenTrackerCurrentUsage:
    """Test current usage functionality."""
    
    def test_get_current_usage(self):
        """Test getting current usage."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            mock_tokenizer_instance.encode.return_value = [1]
            
            tracker = TokenTracker()
            
            messages = [Message(role="user", content="Hello")]
            
            usage = tracker.get_current_usage(
                messages=messages,
                system_prompt="System",
                context_docs=["Doc"]
            )
            
            assert isinstance(usage, TokenUsage)
            assert usage.current >= 0
            assert usage.max == 1500
            assert usage.percentage >= 0
    
    def test_get_current_usage_none_defaults(self):
        """Test current usage with None defaults."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            tracker = TokenTracker()
            
            usage = tracker.get_current_usage()
            
            assert usage.current == 0
            assert usage.max == 1500


@pytest.mark.unit
class TestTokenTrackerLimitChecking:
    """Test limit checking functionality."""
    
    def test_is_approaching_limit_default_threshold(self):
        """Test approaching limit with default threshold."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            tracker = TokenTracker()
            
            tracker._current_usage = TokenUsage(
                current=1300,
                max=1500,
                percentage=86.67,
                breakdown=TokenBreakdown(100, 200, 1000, 1300, 86.67)
            )
            
            assert tracker.is_approaching_limit() is True
    
    def test_is_approaching_limit_custom_threshold(self):
        """Test approaching limit with custom threshold."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            tracker = TokenTracker()
            
            tracker._current_usage = TokenUsage(
                current=1000,
                max=1500,
                percentage=66.67,
                breakdown=TokenBreakdown(100, 200, 700, 1000, 66.67)
            )
            
            assert tracker.is_approaching_limit(threshold=0.5) is True
            assert tracker.is_approaching_limit(threshold=0.8) is False
    
    def test_is_approaching_limit_no_usage(self):
        """Test approaching limit with no usage data."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            tracker = TokenTracker()
            
            assert tracker.is_approaching_limit() is False


@pytest.mark.unit
class TestTokenTrackerWarningLevel:
    """Test warning level functionality."""
    
    def test_get_token_warning_level_normal(self):
        """Test warning level normal."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            tracker = TokenTracker()
            
            tracker._current_usage = TokenUsage(
                current=500,
                max=1500,
                percentage=33.33,
                breakdown=TokenBreakdown(100, 200, 200, 500, 33.33)
            )
            
            assert tracker.get_token_warning_level() == "normal"
    
    def test_get_token_warning_level_warning(self):
        """Test warning level warning."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            tracker = TokenTracker()
            
            tracker._current_usage = TokenUsage(
                current=1300,
                max=1500,
                percentage=86.67,
                breakdown=TokenBreakdown(100, 200, 1000, 1300, 86.67)
            )
            
            assert tracker.get_token_warning_level() == "warning"
    
    def test_get_token_warning_level_critical(self):
        """Test warning level critical."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            tracker = TokenTracker()
            
            tracker._current_usage = TokenUsage(
                current=1400,
                max=1500,
                percentage=93.33,
                breakdown=TokenBreakdown(100, 200, 1100, 1400, 93.33)
            )
            
            assert tracker.get_token_warning_level() == "critical"
    
    def test_get_token_warning_level_no_usage(self):
        """Test warning level with no usage data."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            tracker = TokenTracker()
            
            assert tracker.get_token_warning_level() == "normal"


@pytest.mark.unit
class TestTokenTrackerEstimateResponse:
    """Test response token estimation."""
    
    def test_estimate_response_tokens(self):
        """Test estimating available response tokens."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
            
            tracker = TokenTracker()
            
            available = tracker.estimate_response_tokens("Hello world test")
            
            assert available >= 0
    
    def test_estimate_response_tokens_with_max(self):
        """Test estimating with max new tokens."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            mock_tokenizer_instance.encode.return_value = [1, 2]
            
            tracker = TokenTracker()
            
            available = tracker.estimate_response_tokens("Hello", max_new_tokens=100)
            
            assert available <= 100
    
    def test_estimate_response_tokens_large_prompt(self):
        """Test estimation when prompt exceeds context."""
        with patch('core.token_tracker.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            mock_tokenizer_instance.encode.return_value = [1] * 2000
            
            tracker = TokenTracker()
            
            available = tracker.estimate_response_tokens("Large text" * 500)
            
            assert available == 0


@pytest.mark.unit
class TestGetTokenTracker:
    """Test get_token_tracker singleton."""
    
    def test_singleton(self):
        """Test get_token_tracker returns singleton."""
        import core.token_tracker as tt_module
        tt_module._token_tracker = None
        
        tracker1 = get_token_tracker()
        tracker2 = get_token_tracker()
        
        assert tracker1 is tracker2
