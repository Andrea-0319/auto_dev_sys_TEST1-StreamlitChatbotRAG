"""Reasoning extraction and formatting."""

import re
from typing import Tuple, List, Optional

from utils.logger import setup_logger

logger = setup_logger(__name__)


class ReasoningExtractor:
    """Extract and format model reasoning."""
    
    def __init__(self):
        self.reasoning_patterns = [
            # Standard reasoning tags
            r'(?:REASONING|Thinking|Analysis):\s*(.+?)(?=ANSWER:|Answer:|$)',
            # Step-by-step markers
            r'(?:Step \d+[.:]|\d+\.)\s*(.+?)(?=Step \d+[.:]|\d+\.|Answer:|$)',
            # Think tags (used by some models)
            r'<think>(.+?)</think>',
            r'<<(.+?)>>',
        ]
    
    def extract(self, text: str) -> Tuple[str, str]:
        """Extract reasoning and answer from model output.
        
        Args:
            text: Full model output.
            
        Returns:
            Tuple of (reasoning, answer).
        """
        text = text.strip()
        
        # Try to find explicit reasoning section
        reasoning, answer = self._extract_explicit_sections(text)
        if reasoning:
            return reasoning, answer
        
        # Try to detect step-by-step reasoning
        reasoning, answer = self._extract_step_reasoning(text)
        if reasoning:
            return reasoning, answer
        
        # Default: treat first part as reasoning if it contains analysis words
        reasoning, answer = self._extract_heuristic(text)
        
        return reasoning, answer
    
    def _extract_explicit_sections(self, text: str) -> Tuple[str, str]:
        """Extract reasoning from explicit sections."""
        # Look for REASONING/ANSWER pattern
        match = re.search(
            r'(?:REASONING|Thinking|Analysis|Reasoning Process):\s*(.+?)(?:ANSWER|Answer|Response):\s*(.+)',
            text,
            re.DOTALL | re.IGNORECASE
        )
        
        if match:
            reasoning = match.group(1).strip()
            answer = match.group(2).strip()
            return reasoning, answer
        
        # Look for think tags
        match = re.search(r'<think>(.+?)</think>\s*(.+)', text, re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            answer = match.group(2).strip()
            return reasoning, answer
        
        return "", text
    
    def _extract_step_reasoning(self, text: str) -> Tuple[str, str]:
        """Extract step-by-step reasoning."""
        steps = []
        answer = ""
        
        # Find all steps
        step_pattern = r'(?:^|\n)(?:Step\s*\d+[.:]?\s*)(.+?)(?=\n(?:Step\s*\d+[.:]?|Answer:|$))'
        matches = re.finditer(step_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            step_text = match.group(1).strip()
            if step_text:
                steps.append(step_text)
        
        if steps:
            # Look for answer after steps
            answer_match = re.search(r'(?:Answer:|Therefore,|In conclusion,|So,)(.+)', text, re.DOTALL | re.IGNORECASE)
            if answer_match:
                answer = answer_match.group(1).strip()
            else:
                # Use remaining text after last step
                last_step = steps[-1]
                idx = text.rfind(last_step)
                if idx > 0:
                    remaining = text[idx + len(last_step):].strip()
                    if remaining:
                        answer = remaining
            
            reasoning = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(steps))
            return reasoning, answer if answer else text
        
        return "", text
    
    def _extract_heuristic(self, text: str) -> Tuple[str, str]:
        """Heuristic extraction based on content analysis."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) <= 2:
            return "", text
        
        # Look for reasoning indicators
        reasoning_indicators = [
            r'(?:based on|according to|from the context)',
            r'(?:first|second|third|finally|therefore|thus|because)',
            r'(?:the text mentions|this suggests|this indicates)',
            r'(?:let me|I need to|I should|I will)',
        ]
        
        reasoning_sentences = []
        answer_sentences = []
        in_reasoning = True
        
        for sentence in sentences:
            if in_reasoning:
                # Check if this looks like reasoning
                is_reasoning = any(
                    re.search(pattern, sentence, re.IGNORECASE) 
                    for pattern in reasoning_indicators
                )
                
                if is_reasoning or len(reasoning_sentences) < 2:
                    reasoning_sentences.append(sentence)
                else:
                    answer_sentences.append(sentence)
                    in_reasoning = False
            else:
                answer_sentences.append(sentence)
        
        # If we found reasoning, separate it
        if len(reasoning_sentences) >= 2 and len(answer_sentences) >= 1:
            reasoning = " ".join(reasoning_sentences)
            answer = " ".join(answer_sentences)
            return reasoning, answer
        
        return "", text
    
    def format_reasoning(self, reasoning: str) -> List[str]:
        """Format reasoning into a list of steps.
        
        Args:
            reasoning: Raw reasoning text.
            
        Returns:
            List of reasoning steps.
        """
        if not reasoning:
            return []
        
        # Split by newlines and step markers
        steps = re.split(r'\n+|(?:Step\s*\d+[.:]\s*)', reasoning)
        steps = [s.strip() for s in steps if s.strip()]
        
        return steps
    
    def get_confidence(self, reasoning: str, answer: str) -> Optional[float]:
        """Estimate confidence based on reasoning quality.
        
        Args:
            reasoning: Reasoning text.
            answer: Answer text.
            
        Returns:
            Confidence score between 0 and 1, or None.
        """
        if not reasoning or not answer:
            return None
        
        score = 0.5  # Base score
        
        # More reasoning steps = higher confidence
        steps = self.format_reasoning(reasoning)
        if len(steps) >= 3:
            score += 0.2
        elif len(steps) >= 1:
            score += 0.1
        
        # Longer answer = more detailed = higher confidence
        if len(answer) > 100:
            score += 0.1
        
        # Contains citations = higher confidence
        if re.search(r'\[.*?\]|\(.*?\)', answer):
            score += 0.1
        
        return min(score, 1.0)


def extract_reasoning(text: str) -> Tuple[str, str]:
    """Convenience function to extract reasoning.
    
    Args:
        text: Model output text.
        
    Returns:
        Tuple of (reasoning, answer).
    """
    extractor = ReasoningExtractor()
    return extractor.extract(text)
