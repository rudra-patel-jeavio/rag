"""
Database models and connection management for the RAG application.
"""

import os
import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, LargeBinary, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class PDFDocument(Base):
    """Model for storing PDF documents with metadata."""
    __tablename__ = 'pdf_documents'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    folder_name = Column(String(255), nullable=False, index=True)
    file_name = Column(String(255), nullable=False)
    pdf_file = Column(LargeBinary, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to chunks
    chunks = relationship("PDFChunk", back_populates="document", cascade="all, delete-orphan")

class PDFChunk(Base):
    """Model for storing PDF text chunks with embeddings."""
    __tablename__ = 'pdf_chunks'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('pdf_documents.id'), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=False)  # OpenAI embedding dimension
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to document
    document = relationship("PDFDocument", back_populates="chunks")

class ChatSession(Base):
    """Model for storing chat sessions."""
    __tablename__ = 'chat_sessions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    folder_name = Column(String(255), nullable=True)  # None for global chat
    model_used = Column(String(100), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to messages
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    """Model for storing individual chat messages."""
    __tablename__ = 'chat_messages'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('chat_sessions.id'), nullable=False)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to session
    session = relationship("ChatSession", back_populates="messages")

class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all tables and enable pgvector extension."""
        try:
            # Enable pgvector extension
            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    def close_session(self, session: Session):
        """Close a database session."""
        session.close()

# Database operations
class PDFDocumentService:
    """Service for PDF document operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def create_document(self, folder_name: str, file_name: str, pdf_data: bytes) -> PDFDocument:
        """Create a new PDF document record."""
        session = self.db_manager.get_session()
        try:
            document = PDFDocument(
                folder_name=folder_name,
                file_name=file_name,
                pdf_file=pdf_data
            )
            session.add(document)
            session.commit()
            session.refresh(document)
            return document
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating document: {e}")
            raise
        finally:
            self.db_manager.close_session(session)
    
    def get_documents_by_folder(self, folder_name: str) -> List[PDFDocument]:
        """Get all documents in a folder."""
        session = self.db_manager.get_session()
        try:
            documents = session.query(PDFDocument).filter(
                PDFDocument.folder_name == folder_name
            ).all()
            return documents
        finally:
            self.db_manager.close_session(session)
    
    def get_all_folders(self) -> List[str]:
        """Get all unique folder names."""
        session = self.db_manager.get_session()
        try:
            folders = session.query(PDFDocument.folder_name).distinct().all()
            return [folder[0] for folder in folders]
        finally:
            self.db_manager.close_session(session)

class PDFChunkService:
    """Service for PDF chunk operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def create_chunks(self, document_id: uuid.UUID, chunks_data: List[dict]):
        """Create multiple chunks for a document."""
        session = self.db_manager.get_session()
        try:
            chunks = []
            for chunk_data in chunks_data:
                chunk = PDFChunk(
                    document_id=document_id,
                    chunk_index=chunk_data['index'],
                    chunk_text=chunk_data['text'],
                    embedding=chunk_data['embedding']
                )
                chunks.append(chunk)
            
            session.add_all(chunks)
            session.commit()
            return chunks
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating chunks: {e}")
            raise
        finally:
            self.db_manager.close_session(session)
    
    def search_similar_chunks(self, query_embedding: List[float], folder_name: Optional[str] = None, limit: int = 5) -> List[dict]:
        """Search for similar chunks using cosine similarity."""
        session = self.db_manager.get_session()
        try:
            query = session.query(
                PDFChunk.chunk_text,
                PDFDocument.file_name,
                PDFDocument.folder_name,
                PDFChunk.embedding.cosine_distance(query_embedding).label('distance')
            ).join(PDFDocument)
            
            if folder_name:
                query = query.filter(PDFDocument.folder_name == folder_name)
            
            results = query.order_by('distance').limit(limit).all()
            
            return [
                {
                    'text': result.chunk_text,
                    'file_name': result.file_name,
                    'folder_name': result.folder_name,
                    'distance': result.distance
                }
                for result in results
            ]
        finally:
            self.db_manager.close_session(session)

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