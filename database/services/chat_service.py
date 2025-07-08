"""
Chat service for the RAG application.
"""

import uuid
import logging
from typing import List, Optional
from ..models import ChatSession, ChatMessage
from ..connection import DatabaseManager

logger = logging.getLogger(__name__)

class ChatService:
    """Service for chat operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def create_session(self, folder_name: Optional[str], model_used: str) -> ChatSession:
        """Create a new chat session."""
        session = self.db_manager.get_session()
        try:
            chat_session = ChatSession(
                folder_name=folder_name,
                model_used=model_used
            )
            session.add(chat_session)
            session.commit()
            session.refresh(chat_session)
            return chat_session
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating chat session: {e}")
            raise
        finally:
            self.db_manager.close_session(session)
    
    def add_message(self, session_id: uuid.UUID, role: str, message: str):
        """Add a message to a chat session."""
        session = self.db_manager.get_session()
        try:
            chat_message = ChatMessage(
                session_id=session_id,
                role=role,
                message=message
            )
            session.add(chat_message)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding message: {e}")
            raise
        finally:
            self.db_manager.close_session(session)
    
    def get_session_messages(self, session_id: uuid.UUID) -> List[ChatMessage]:
        """Get all messages for a session."""
        session = self.db_manager.get_session()
        try:
            messages = session.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).order_by(ChatMessage.timestamp).all()
            return messages
        finally:
            self.db_manager.close_session(session) 