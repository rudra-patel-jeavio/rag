"""
Database models and connection management for the RAG application.
This file provides backward compatibility by importing from the modular database package.
"""

# Import everything from the modular database package
from database import (
    DatabaseManager,
    PDFDocument,
    PDFChunk,
    ChatSession,
    ChatMessage,
    Base,
    PDFDocumentService,
    PDFChunkService,
    ChatService
)

# Re-export for backward compatibility
__all__ = [
    'DatabaseManager',
    'PDFDocument',
    'PDFChunk',
    'ChatSession',
    'ChatMessage',
    'Base',
    'PDFDocumentService',
    'PDFChunkService',
    'ChatService'
] 