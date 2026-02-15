"""Query-response workflow orchestration."""

from typing import Dict, Generator, List, Optional, Tuple

from config import settings
from utils.logger import setup_logger
from utils.validators import validate_user_input
from core.session import SessionManager, get_session_manager
from rag.engine import RAGEngine
from model.handler import ModelHandler

logger = setup_logger(__name__)


class WorkflowEngine:
    """Orchestrate the query-response workflow."""
    
    def __init__(
        self,
        session_manager: SessionManager = None,
        rag_engine: RAGEngine = None,
        model_handler: ModelHandler = None
    ):
        """Initialize workflow engine.
        
        Args:
            session_manager: Session manager instance.
            rag_engine: RAG engine instance.
            model_handler: Model handler instance.
        """
        self.session = session_manager or get_session_manager()
        self.rag = rag_engine or RAGEngine()
        self.model = model_handler or ModelHandler()
        
        logger.info("WorkflowEngine initialized")
    
    def process_query(
        self,
        query: str,
        stream: bool = True
    ) -> Generator[Dict, None, None]:
        """Process a user query through the full workflow.
        
        Args:
            query: User query.
            stream: Whether to stream the response.
            
        Yields:
            Status updates and results.
        """
        # Validate input
        is_valid, error_msg = validate_user_input(query)
        if not is_valid:
            yield {
                "type": "error",
                "message": error_msg
            }
            return
        
        logger.info(f"Processing query: {query[:50]}...")
        
        # Add user message to history
        self.session.add_user_message(query)
        
        yield {
            "type": "status",
            "message": "Retrieving relevant documents..."
        }
        
        # Retrieve context
        try:
            context = self.rag.get_context_string(query)
            chunks = self.rag.retrieve(query)
            
            sources = list(set(c["document_name"] for c in chunks)) if chunks else []
            
            yield {
                "type": "retrieval",
                "chunks": chunks,
                "sources": sources,
                "context": context[:500] + "..." if len(context) > 500 else context
            }
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            context = ""
            chunks = []
            sources = []
        
        yield {
            "type": "status",
            "message": "Generating response..."
        }
        
        # Build prompt
        history = self.session.get_conversation_history(max_turns=settings.MAX_CHAT_HISTORY)
        prompt = self.model._prompt_builder.chat_prompt(query, context, history)
        
        # Generate response
        try:
            if stream:
                # Stream response
                full_response = ""
                for token in self.model.generate_stream(prompt):
                    full_response += token
                    yield {
                        "type": "token",
                        "token": token,
                        "partial_response": full_response
                    }
                
                # Extract reasoning
                reasoning, answer = self.model.extract_reasoning(full_response)
                
                # Add to session
                self.session.add_assistant_message(answer, reasoning, sources)
                
                yield {
                    "type": "complete",
                    "response": answer,
                    "reasoning": reasoning,
                    "sources": sources,
                    "full_response": full_response
                }
                
            else:
                # Non-streaming response
                full_response = self.model.generate(prompt)
                reasoning, answer = self.model.extract_reasoning(full_response)
                
                self.session.add_assistant_message(answer, reasoning, sources)
                
                yield {
                    "type": "complete",
                    "response": answer,
                    "reasoning": reasoning,
                    "sources": sources,
                    "full_response": full_response
                }
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            yield {
                "type": "error",
                "message": f"Failed to generate response: {str(e)}"
            }
    
    def stop_generation(self) -> None:
        """Stop ongoing generation."""
        self.model.stop_generation()
        logger.info("Generation stopped by user")
    
    def upload_document(self, file_path: str, file_content: bytes = None) -> Tuple[bool, str]:
        """Upload and process a document.
        
        Args:
            file_path: Path to file.
            file_content: Optional file content.
            
        Returns:
            Tuple of (success, message).
        """
        success, message = self.rag.add_document(file_path, file_content)
        
        if success:
            # Get document ID from the message
            # Note: We'd need to modify RAGEngine to return the doc_id
            logger.info(f"Document uploaded: {message}")
        else:
            logger.error(f"Document upload failed: {message}")
        
        return success, message
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a document.
        
        Args:
            doc_id: Document ID.
            
        Returns:
            True if removed.
        """
        return self.rag.remove_document(doc_id)
    
    def get_document_list(self) -> List[Dict]:
        """Get list of uploaded documents."""
        return self.rag.get_document_list()
    
    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self.session.clear_conversation()
    
    def clear_all_documents(self) -> bool:
        """Clear all documents."""
        return self.rag.clear_all()
    
    def get_stats(self) -> Dict:
        """Get workflow statistics."""
        return {
            "session": self.session.get_session_info(),
            "rag": self.rag.get_stats(),
            "model": self.model.get_model_info(),
        }


def create_workflow(
    session_manager: SessionManager = None,
    rag_engine: RAGEngine = None,
    model_handler: ModelHandler = None
) -> WorkflowEngine:
    """Create a workflow engine.
    
    Args:
        session_manager: Optional session manager.
        rag_engine: Optional RAG engine.
        model_handler: Optional model handler.
        
    Returns:
        WorkflowEngine instance.
    """
    return WorkflowEngine(session_manager, rag_engine, model_handler)
