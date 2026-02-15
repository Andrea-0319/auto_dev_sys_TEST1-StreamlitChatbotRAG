"""Export service for multiple export formats."""

import json
from datetime import datetime
from typing import List, Optional

from core.session import Message
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ExportService:
    """Handle multiple export formats with metadata."""
    
    def export_markdown(
        self,
        messages: List[Message],
        include_metadata: bool = True
    ) -> str:
        """Export conversation as Markdown.
        
        Args:
            messages: Messages to export.
            include_metadata: Include timestamps and metadata.
            
        Returns:
            Markdown formatted string.
        """
        lines = ["# Chat History\n"]
        
        if include_metadata:
            lines.append(f"*Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        lines.append("")
        
        for i, msg in enumerate(messages):
            role_icon = "🧑" if msg.role == "user" else "🤖" if msg.role == "assistant" else "⚙️"
            
            lines.append(f"## {role_icon} {msg.role.title()}")
            
            if include_metadata:
                if msg.timestamp:
                    timestamp = msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    timestamp = "N/A"
                lines.append(f"*Timestamp: {timestamp}*")
                
                if msg.id:
                    lines.append(f"*ID: {msg.id}*")
            
            if msg.is_pinned:
                lines.append("*📌 Pinned*")
            
            if msg.feedback:
                feedback_icon = "👍" if msg.feedback == "positive" else "👎"
                lines.append(f"*{feedback_icon} Feedback: {msg.feedback}*")
            
            lines.append("")
            lines.append(msg.content)
            
            if msg.reasoning and include_metadata:
                lines.append("")
                lines.append(f"> *Reasoning: {msg.reasoning[:200]}...*")
            
            if msg.sources:
                lines.append("")
                lines.append(f"> *Sources: {', '.join(msg.sources)}*")
            
            lines.append("\n---\n")
        
        return "\n".join(lines)
    
    def export_json(
        self,
        messages: List[Message],
        include_metadata: bool = True
    ) -> str:
        """Export conversation as JSON.
        
        Args:
            messages: Messages to export.
            include_metadata: Include all metadata.
            
        Returns:
            JSON formatted string.
        """
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "message_count": len(messages),
            "messages": []
        }
        
        for msg in messages:
            msg_dict = {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            
            if include_metadata:
                if msg.reasoning:
                    msg_dict["reasoning"] = msg.reasoning
                if msg.sources:
                    msg_dict["sources"] = ", ".join(msg.sources)
                if msg.is_pinned:
                    msg_dict["is_pinned"] = str(msg.is_pinned)
                if msg.feedback:
                    msg_dict["feedback"] = msg.feedback
                if msg.branch_id:
                    msg_dict["branch_id"] = msg.branch_id
                if msg.parent_message_id:
                    msg_dict["parent_message_id"] = msg.parent_message_id
            
            export_data["messages"].append(msg_dict)
        
        return json.dumps(export_data, indent=2)
    
    def export_plain_text(self, messages: List[Message]) -> str:
        """Export conversation as plain text.
        
        Args:
            messages: Messages to export.
            
        Returns:
            Plain text string.
        """
        lines = []
        
        for msg in messages:
            role = msg.role.upper()
            if msg.timestamp:
                timestamp = msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                timestamp = "N/A"
            
            lines.append(f"[{timestamp}] {role}:")
            lines.append(msg.content)
            lines.append("")
        
        return "\n".join(lines)
    
    def export_with_timestamps(
        self,
        messages: List[Message],
        format: str = "markdown"
    ) -> str:
        """Export with timestamps in specified format.
        
        Args:
            messages: Messages to export.
            format: Export format ("markdown", "json", "plain").
            
        Returns:
            Formatted export string.
        """
        if format == "json":
            return self.export_json(messages, include_metadata=True)
        elif format == "plain":
            return self.export_plain_text(messages)
        else:
            return self.export_markdown(messages, include_metadata=True)
    
    def export_with_branch_info(
        self,
        messages: List[Message],
        branches: List
    ) -> str:
        """Export with branch information.
        
        Args:
            messages: Messages to export.
            branches: Branch information.
            
        Returns:
            JSON export with branch data.
        """
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "branches": [
                {
                    "id": b.id,
                    "name": b.name,
                    "created_at": b.created_at.isoformat(),
                    "is_active": b.is_active,
                    "message_count": b.message_count
                }
                for b in branches
            ],
            "messages": []
        }
        
        for msg in messages:
            msg_dict = {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "branch_id": msg.branch_id,
                "parent_message_id": msg.parent_message_id
            }
            
            if msg.reasoning:
                msg_dict["reasoning"] = msg.reasoning
            if msg.sources:
                msg_dict["sources"] = ", ".join(msg.sources)
            
            export_data["messages"].append(msg_dict)
        
        return json.dumps(export_data, indent=2)


_export_service: Optional[ExportService] = None


def get_export_service() -> 'ExportService':
    """Get or create global export service."""
    global _export_service
    if _export_service is None:
        _export_service = ExportService()
    return _export_service
