"""Summarization service for context summarization functionality."""

from dataclasses import dataclass
from typing import List, Optional
from uuid import uuid4

from core.session import Message
from model.handler import ModelHandler
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class SummaryResult:
    """Result of summarization."""
    summary: str
    original_message_ids: List[str]
    new_message: Optional[Message]
    tokens_saved: int


class SummarizationService:
    """Generate context summaries using the LLM model."""
    
    def __init__(self, model_handler: ModelHandler):
        """Initialize summarization service.
        
        Args:
            model_handler: Model handler for generating summaries.
        """
        self._model_handler = model_handler
    
    def summarize_messages(
        self,
        messages: List[Message],
        preserve_recent: int = 4
    ) -> SummaryResult:
        """Summarize older messages while preserving recent ones.
        
        Args:
            messages: List of messages to summarize.
            preserve_recent: Number of recent messages to preserve.
            
        Returns:
            Summary result with condensed content.
        """
        if len(messages) < preserve_recent:
            logger.info("Not enough messages to summarize")
            return SummaryResult(
                summary="",
                original_message_ids=[],
                new_message=None,
                tokens_saved=0
            )
        
        if preserve_recent == 0:
            messages_to_summarize = messages
            messages_to_preserve = []
        else:
            messages_to_summarize = messages[:-preserve_recent]
            messages_to_preserve = messages[-preserve_recent:]
        
        original_ids = [m.id for m in messages_to_summarize]
        
        summary_prompt = self.get_summary_prompt(messages_to_summarize)
        
        try:
            summary_response = self._model_handler.generate(
                prompt=summary_prompt
            )
            
            if isinstance(summary_response, dict):
                summary = summary_response.get("response", "")
            else:
                summary = str(summary_response)
            
            new_message = self.create_summary_message(summary, original_ids)
            
            original_tokens = sum(
                len(m.content) // 4 + len(m.reasoning or "") // 4 + 1
                for m in messages_to_summarize
            )
            summary_tokens = len(summary) // 4 + 1
            tokens_saved = max(0, original_tokens - summary_tokens)
            
            logger.info(f"Summarized {len(messages_to_summarize)} messages, saved ~{tokens_saved} tokens")
            
            return SummaryResult(
                summary=summary,
                original_message_ids=original_ids,
                new_message=new_message,
                tokens_saved=tokens_saved
            )
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return SummaryResult(
                summary="",
                original_message_ids=[],
                new_message=None,
                tokens_saved=0
            )
    
    def create_summary_message(
        self,
        summary: str,
        original_message_ids: List[str]
    ) -> Message:
        """Create a summary message.
        
        Args:
            summary: Summary text.
            original_message_ids: IDs of messages summarized.
            
        Returns:
            Summary message.
        """
        return Message(
            role="system",
            content=f"[Previous conversation summary: {summary}]",
            sources=[f"summarized:{mid}" for mid in original_message_ids]
        )
    
    def get_summary_prompt(self, messages: List[Message]) -> str:
        """Generate prompt for summarization.
        
        Args:
            messages: Messages to summarize.
            
        Returns:
            Summary prompt.
        """
        conversation_text = []
        
        for msg in messages:
            role = msg.role.title()
            content = msg.content
            conversation_text.append(f"{role}: {content}")
        
        conversation_str = "\n\n".join(conversation_text)
        
        prompt = f"""Please provide a concise summary of the following conversation. 
Focus on the key points, questions asked, and important information shared.

CONVERSATION:
{conversation_str}

SUMMARY (be brief, 2-4 sentences):"""
        
        return prompt


_summarization_service: Optional[SummarizationService] = None


def get_summarization_service(model_handler: Optional[ModelHandler] = None) -> Optional['SummarizationService']:
    """Get or create global summarization service.
    
    Args:
        model_handler: Optional model handler.
        
    Returns:
        Summarization service instance.
    """
    global _summarization_service
    
    if model_handler is None:
        try:
            from model.handler import get_model_handler
            model_handler = get_model_handler()
        except ImportError:
            pass
    
    if _summarization_service is None and model_handler is not None:
        _summarization_service = SummarizationService(model_handler)
    
    return _summarization_service
