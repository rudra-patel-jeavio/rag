"""
SQLAlchemy models for the RAG application.
Database-agnostic models that work with both PostgreSQL and SQLite.
"""

import uuid
import json
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, Integer, LargeBinary, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.types import TypeDecorator, TEXT

# Try to import PostgreSQL-specific types, fallback to generic types
try:
    from sqlalchemy.dialects.postgresql import UUID
    from pgvector.sqlalchemy import Vector as PGVector
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    # Fallback UUID type for SQLite
    class UUID(TypeDecorator):
        impl = String(36)
        cache_ok = True
        
        def process_bind_param(self, value, dialect):
            if value is None:
                return value
            elif dialect.name == 'postgresql':
                return str(value)
            else:
                return str(value)
        
        def process_result_value(self, value, dialect):
            if value is None:
                return value
            else:
                return uuid.UUID(value)

# Custom Vector type that works with both PostgreSQL and SQLite
class Vector(TypeDecorator):
    impl = TEXT  # Default implementation
    cache_ok = True
    
    def __init__(self, dimensions=None):
        super().__init__()
        self.dimensions = dimensions
        # Override impl for PostgreSQL if pgvector is available
        if POSTGRES_AVAILABLE:
            self.impl = PGVector(dimensions)
    
    def load_dialect_impl(self, dialect):
        """Load the appropriate implementation based on dialect."""
        if dialect.name == 'postgresql' and POSTGRES_AVAILABLE:
            return dialect.type_descriptor(PGVector(self.dimensions))
        else:
            return dialect.type_descriptor(TEXT())
    
    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        
        if dialect.name == 'postgresql' and POSTGRES_AVAILABLE:
            # For PostgreSQL with pgvector, return the list directly
            if isinstance(value, list):
                return value
            elif hasattr(value, 'tolist'):
                return value.tolist()
            else:
                return list(value)
        else:
            # Store as JSON string for SQLite
            if isinstance(value, list):
                return json.dumps(value)
            elif hasattr(value, 'tolist'):
                return json.dumps(value.tolist())
            else:
                return json.dumps(list(value))
    
    def process_result_value(self, value, dialect):
        if value is None:
            return value
        
        if dialect.name == 'postgresql' and POSTGRES_AVAILABLE:
            # For PostgreSQL with pgvector, value should already be a list
            if isinstance(value, list):
                return value
            elif hasattr(value, 'tolist'):
                return value.tolist()
            else:
                return list(value)
        else:
            # Parse JSON for SQLite
            if isinstance(value, str):
                return json.loads(value)
            else:
                return value
    
    class comparator_factory(TypeDecorator.Comparator):
        def cosine_distance(self, other):
            # For PostgreSQL with pgvector, use the native operator
            if POSTGRES_AVAILABLE:
                from sqlalchemy.types import Float
                return self.op('<=>', return_type=Float)(other)
            else:
                # For SQLite, this will be handled in the service layer
                # Just return a dummy expression that won't be used
                from sqlalchemy.sql import literal
                return literal(0.0)

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
    embedding = Column(Vector(384), nullable=False)  # Sentence transformer embedding dimension
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