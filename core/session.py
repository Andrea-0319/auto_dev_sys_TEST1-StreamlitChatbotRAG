"""Session state management."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Message:
    """Represents a conversation message."""
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    reasoning: Optional[str] = None
    sources: Optional[List[str]] = None
    id: str = field(default_factory=lambda: str(uuid4()))


class SessionManager:
    """Manage conversation and application state."""
    
    def __init__(self):
        """Initialize session manager."""
        self._messages: List[Message] = []
        self._session_id = str(uuid4())
        self._created_at = datetime.now()
        self._document_ids: set = set()
        
        logger.info(f"Session initialized: {self._session_id}")
    
    def add_user_message(self, content: str) -> Message:
        """Add a user message to history.
        
        Args:
            content: Message content.
            
        Returns:
            Created message.
        """
        message = Message(role="user", content=content)
        self._messages.append(message)
        logger.debug(f"Added user message: {content[:50]}...")
        return message
    
    def add_assistant_message(
        self,
        content: str,
        reasoning: Optional[str] = None,
        sources: Optional[List[str]] = None
    ) -> Message:
        """Add an assistant response to history.
        
        Args:
            content: Response content.
            reasoning: Optional reasoning text.
            sources: Optional source document names.
            
        Returns:
            Created message.
        """
        message = Message(
            role="assistant",
            content=content,
            reasoning=reasoning,
            sources=sources
        )
        self._messages.append(message)
        logger.debug(f"Added assistant message: {content[:50]}...")
        return message
    
    def add_system_message(self, content: str) -> Message:
        """Add a system message.
        
        Args:
            content: Message content.
            
        Returns:
            Created message.
        """
        message = Message(role="system", content=content)
        self._messages.append(message)
        return message
    
    def get_messages(self, max_turns: int = None) -> List[Message]:
        """Get conversation messages.
        
        Args:
            max_turns: Maximum number of recent turns to return.
            
        Returns:
            List of messages.
        """
        if max_turns is None:
            return self._messages.copy()
        
        # Get last N turns (a turn is user + assistant pair)
        messages = []
        count = 0
        
        for msg in reversed(self._messages):
            if msg.role in ("user", "assistant"):
                if msg.role == "user":
                    count += 1
                    if count > max_turns:
                        break
            messages.insert(0, msg)
        
        return messages
    
    def get_conversation_history(self, max_turns: int = None) -> List[Dict]:
        """Get conversation history as dictionaries.
        
        Args:
            max_turns: Maximum number of recent turns.
            
        Returns:
            List of message dictionaries.
        """
        messages = self.get_messages(max_turns)
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "reasoning": msg.reasoning,
                "sources": msg.sources,
            }
            for msg in messages
        ]
    
    def get_formatted_history(self, for_model: bool = True, max_turns: int = None) -> str:
        """Get history formatted for display or model.
        
        Args:
            for_model: Format for model input (vs display).
            max_turns: Maximum turns to include.
            
        Returns:
            Formatted history string.
        """
        messages = self.get_messages(max_turns)
        
        if for_model:
            # Format for model (includes only user/assistant)
            parts = []
            for msg in messages:
                if msg.role == "user":
                    parts.append(f"User: {msg.content}")
                elif msg.role == "assistant":
                    parts.append(f"Assistant: {msg.content}")
            return "\n".join(parts)
        else:
            # Format for display
            parts = []
            for msg in messages:
                prefix = "🧑" if msg.role == "user" else "🤖" if msg.role == "assistant" else "⚙️"
                parts.append(f"{prefix} [{msg.timestamp.strftime('%H:%M:%S')}] {msg.content}")
            return "\n".join(parts)
    
    def clear_conversation(self) -> None:
        """Clear all conversation messages."""
        self._messages = []
        logger.info("Conversation cleared")
    
    def get_last_message(self) -> Optional[Message]:
        """Get the most recent message."""
        return self._messages[-1] if self._messages else None
    
    def update_last_message(self, content: str = None, reasoning: str = None) -> bool:
        """Update the last assistant message.
        
        Args:
            content: New content.
            reasoning: New reasoning.
            
        Returns:
            True if updated.
        """
        if not self._messages:
            return False
        
        last_msg = self._messages[-1]
        if last_msg.role != "assistant":
            return False
        
        if content is not None:
            last_msg.content = content
        if reasoning is not None:
            last_msg.reasoning = reasoning
        
        return True
    
    def register_document(self, doc_id: str) -> None:
        """Register a document as part of this session."""
        self._document_ids.add(doc_id)
    
    def unregister_document(self, doc_id: str) -> None:
        """Unregister a document."""
        self._document_ids.discard(doc_id)
    
    def get_session_info(self) -> Dict:
        """Get session information."""
        return {
            "session_id": self._session_id,
            "created_at": self._created_at.isoformat(),
            "message_count": len(self._messages),
            "document_count": len(self._document_ids),
            "duration_minutes": (datetime.now() - self._created_at).total_seconds() / 60,
        }
    
    @property
    def message_count(self) -> int:
        """Get total message count."""
        return len(self._messages)
    
    @property
    def is_empty(self) -> bool:
        """Check if conversation is empty."""
        return len(self._messages) == 0
    
    def export_conversation(self) -> Dict:
        """Export conversation for saving."""
        return {
            "session_id": self._session_id,
            "created_at": self._created_at.isoformat(),
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "reasoning": msg.reasoning,
                    "sources": msg.sources,
                }
                for msg in self._messages
            ],
            "document_ids": list(self._document_ids),
        }
    
    def import_conversation(self, data: Dict) -> bool:
        """Import conversation from saved data.
        
        Args:
            data: Conversation data.
            
        Returns:
            True if successful.
        """
        try:
            self._messages = []
            for msg_data in data.get("messages", []):
                msg = Message(
                    role=msg_data["role"],
                    content=msg_data["content"],
                    timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                    reasoning=msg_data.get("reasoning"),
                    sources=msg_data.get("sources"),
                    id=msg_data.get("id", str(uuid4())),
                )
                self._messages.append(msg)
            
            self._document_ids = set(data.get("document_ids", []))
            logger.info(f"Imported {len(self._messages)} messages")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import conversation: {e}")
            return False


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
