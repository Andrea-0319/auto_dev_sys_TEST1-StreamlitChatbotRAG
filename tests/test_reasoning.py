"""Unit tests for model.reasoning module."""

import pytest
from model.reasoning import ReasoningExtractor, extract_reasoning


@pytest.mark.unit
class TestReasoningExtractorInitialization:
    """Test ReasoningExtractor initialization."""
    
    def test_initialization(self):
        """Test ReasoningExtractor initialization."""
        extractor = ReasoningExtractor()
        
        assert hasattr(extractor, 'reasoning_patterns')
        assert len(extractor.reasoning_patterns) > 0


@pytest.mark.unit
class TestExtractExplicitSections:
    """Test explicit section extraction."""
    
    def test_reasoning_answer_format(self):
        """Test REASONING/ANSWER format."""
        text = """REASONING:
        This is my reasoning process.
        I analyzed the context carefully.
        
        ANSWER:
        This is the final answer."""
        
        extractor = ReasoningExtractor()
        reasoning, answer = extractor.extract(text)
        
        assert "reasoning process" in reasoning.lower()
        assert "final answer" in answer.lower()
    
    def test_thinking_answer_format(self):
        """Test Thinking/Answer format."""
        text = """Thinking: First I considered the options.
        Then I made a decision.
        
        Answer: The best choice is option A."""
        
        extractor = ReasoningExtractor()
        reasoning, answer = extractor.extract(text)
        
        assert "considered the options" in reasoning.lower()
        assert "option A" in answer
    
    def test_think_tags(self):
        """Test <think> tags."""
        text = """<think>
        Step 1: Analyze the problem
        Step 2: Find the solution
        </think>
        The solution is clear."""
        
        extractor = ReasoningExtractor()
        reasoning, answer = extractor.extract(text)
        
        assert "analyze the problem" in reasoning.lower()
        assert "solution is clear" in answer.lower()


@pytest.mark.unit
class TestExtractStepReasoning:
    """Test step-by-step reasoning extraction."""
    
    def test_step_format(self):
        """Test Step N format."""
        text = """Step 1: Identify the key facts
        Step 2: Analyze relationships
        Step 3: Draw conclusions
        
        Answer: Based on this analysis, the answer is 42."""
        
        extractor = ReasoningExtractor()
        reasoning, answer = extractor.extract(text)
        
        assert "Step 1:" in reasoning
        assert "Step 2:" in reasoning
        assert "Step 3:" in reasoning
        assert "42" in answer
    
    def test_numbered_list_format(self):
        """Test numbered list format."""
        text = """1. First, consider the input
        2. Then process the data
        3. Finally, output the result
        
        Therefore, the result is correct."""
        
        extractor = ReasoningExtractor()
        reasoning, answer = extractor.extract(text)
        
        assert len(reasoning) > 0
        assert "result is correct" in answer.lower()


@pytest.mark.unit
class TestExtractHeuristic:
    """Test heuristic reasoning extraction."""
    
    def test_reasoning_indicators(self):
        """Test detection of reasoning indicators."""
        text = """Based on the context provided, I can see that 
        the first step is to analyze the data. Then, I need to 
        process it according to the rules. Finally, I should 
        present the results clearly.
        
        The answer is that this approach works well."""
        
        extractor = ReasoningExtractor()
        reasoning, answer = extractor.extract(text)
        
        assert len(reasoning) > 0
        assert "answer is" in answer.lower()
    
    def test_short_text_no_split(self):
        """Test short text without split."""
        text = "This is a short response."
        
        extractor = ReasoningExtractor()
        reasoning, answer = extractor.extract(text)
        
        assert reasoning == ""
        assert answer == text


@pytest.mark.unit
class TestFormatReasoning:
    """Test reasoning formatting."""
    
    def test_format_reasoning_steps(self):
        """Test formatting reasoning into steps."""
        extractor = ReasoningExtractor()
        reasoning = "Step 1: Do this\nStep 2: Do that\nStep 3: Finish"
        
        steps = extractor.format_reasoning(reasoning)
        
        assert len(steps) >= 3
    
    def test_format_empty_reasoning(self):
        """Test formatting empty reasoning."""
        extractor = ReasoningExtractor()
        
        steps = extractor.format_reasoning("")
        
        assert steps == []
    
    def test_format_single_step(self):
        """Test formatting single step reasoning."""
        extractor = ReasoningExtractor()
        reasoning = "This is a single thought process."
        
        steps = extractor.format_reasoning(reasoning)
        
        assert len(steps) == 1


@pytest.mark.unit
class TestGetConfidence:
    """Test confidence score calculation."""
    
    def test_confidence_with_multiple_steps(self):
        """Test confidence with multiple steps."""
        extractor = ReasoningExtractor()
        reasoning = "Step 1: A\nStep 2: B\nStep 3: C"
        answer = "This is a detailed answer with explanation."
        
        confidence = extractor.get_confidence(reasoning, answer)
        
        assert confidence is not None
        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be higher with multiple steps
    
    def test_confidence_with_citations(self):
        """Test confidence with citations."""
        extractor = ReasoningExtractor()
        reasoning = "Step 1: Analyze"
        answer = "According to [1] and (2), the result is valid."
        
        confidence = extractor.get_confidence(reasoning, answer)
        
        assert confidence is not None
        assert confidence > 0.5
    
    def test_confidence_empty(self):
        """Test confidence with empty input."""
        extractor = ReasoningExtractor()
        
        confidence = extractor.get_confidence("", "")
        
        assert confidence is None


@pytest.mark.unit
class TestExtractReasoningFunction:
    """Test extract_reasoning convenience function."""
    
    def test_extract_reasoning_function(self):
        """Test extract_reasoning convenience function."""
        text = """REASONING:
        This is reasoning.
        
        ANSWER:
        This is the answer."""
        
        reasoning, answer = extract_reasoning(text)
        
        assert "reasoning" in reasoning.lower()
        assert "answer" in answer.lower()
