"""Branch manager service for conversation branching functionality."""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from core.session import Branch, Message, SessionManager
from utils.logger import setup_logger

logger = setup_logger(__name__)


class BranchManager:
    """Manage conversation branches."""
    
    MAX_BRANCHES = 10
    
    def __init__(self, session_manager: SessionManager):
        """Initialize branch manager.
        
        Args:
            session_manager: Session manager instance.
        """
        self._session_manager = session_manager
        self._branch_data: dict = {}
    
    def create_branch(self, from_message: Message, name: str = None) -> Optional[Branch]:
        """Create a new branch from a message.
        
        Args:
            from_message: Message to branch from.
            name: Optional branch name.
            
        Returns:
            Created branch or None if max reached.
        """
        if from_message is None:
            logger.error("Cannot create branch from None message")
            return None
        
        branches = self._session_manager.get_all_branches()
        
        if len(branches) >= self.MAX_BRANCHES:
            logger.warning(f"Maximum branches ({self.MAX_BRANCHES}) reached")
            return None
        
        messages = self._session_manager.get_messages()
        
        try:
            msg_index = next(i for i, m in enumerate(messages) if m.id == from_message.id)
        except StopIteration:
            logger.error(f"Message not found: {from_message.id}")
            return None
        
        branch_id = self._session_manager.create_branch(msg_index, name)
        
        if not branch_id:
            return None
        
        branches = self._session_manager.get_all_branches()
        return next((b for b in branches if b.id == branch_id), None)
    
    def list_branches(self) -> List[Branch]:
        """List all branches.
        
        Returns:
            List of branches.
        """
        return self._session_manager.get_all_branches()
    
    def get_branch(self, branch_id: str) -> Optional[Branch]:
        """Get a specific branch.
        
        Args:
            branch_id: Branch ID.
            
        Returns:
            Branch or None.
        """
        branches = self._session_manager.get_all_branches()
        return next((b for b in branches if b.id == branch_id), None)
    
    def delete_branch(self, branch_id: str) -> bool:
        """Delete a branch.
        
        Args:
            branch_id: Branch ID.
            
        Returns:
            True if successful.
        """
        return self._session_manager.delete_branch(branch_id)
    
    def merge_branch(
        self,
        source_branch_id: str,
        target_branch_id: str
    ) -> bool:
        """Merge one branch into another.
        
        Args:
            source_branch_id: Source branch to merge from.
            target_branch_id: Target branch to merge into.
            
        Returns:
            True if successful.
        """
        if source_branch_id == target_branch_id:
            logger.warning("Cannot merge branch with itself")
            return False
        
        source_branch = self.get_branch(source_branch_id)
        target_branch = self.get_branch(target_branch_id)
        
        if not source_branch or not target_branch:
            logger.warning("Source or target branch not found")
            return False
        
        messages = self._session_manager.get_messages()
        
        source_messages = [
            m for m in messages 
            if m.branch_id == source_branch_id
        ]
        
        for msg in source_messages:
            msg.branch_id = target_branch_id
        
        self._session_manager.delete_branch(source_branch_id)
        
        logger.info(f"Merged branch {source_branch_id} into {target_branch_id}")
        return True
    
    def get_branch_messages(self, branch_id: str) -> List[Message]:
        """Get messages for a specific branch.
        
        Args:
            branch_id: Branch ID.
            
        Returns:
            List of messages in branch.
        """
        messages = self._session_manager.get_messages()
        return [m for m in messages if m.branch_id == branch_id]
    
    def get_branch_tree(self) -> dict:
        """Get branch tree structure.
        
        Returns:
            Dictionary representing branch structure.
        """
        branches = self.list_branches()
        current_id = self._session_manager.get_current_branch_id()
        
        tree = {
            "branches": [
                {
                    "id": b.id,
                    "name": b.name,
                    "created_at": b.created_at.isoformat(),
                    "is_active": b.id == current_id,
                    "message_count": b.message_count
                }
                for b in branches
            ],
            "current_branch": current_id
        }
        
        return tree


_branch_manager: Optional[BranchManager] = None


def get_branch_manager(session_manager: SessionManager = None) -> BranchManager:
    """Get or create global branch manager.
    
    Args:
        session_manager: Optional session manager.
        
    Returns:
        Branch manager instance.
    """
    global _branch_manager
    
    if session_manager is None:
        from core.session import get_session_manager
        session_manager = get_session_manager()
    
    if _branch_manager is None:
        _branch_manager = BranchManager(session_manager)
    
    return _branch_manager
