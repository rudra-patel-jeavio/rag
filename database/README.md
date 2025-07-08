# Database Module

This module contains the database models, connections, and services for the RAG application.

## Structure

```
database/
├── __init__.py          # Package initialization and exports
├── README.md           # This documentation file
├── connection.py       # Database connection and session management
├── models.py          # SQLAlchemy model definitions
└── services/          # Service layer for database operations
    ├── __init__.py    # Services package initialization
    ├── pdf_service.py # PDF document and chunk operations
    └── chat_service.py # Chat session and message operations
```

## Components

### Models (`models.py`)
- **PDFDocument**: Stores PDF files with metadata
- **PDFChunk**: Stores text chunks with embeddings for vector search
- **ChatSession**: Stores chat session information
- **ChatMessage**: Stores individual chat messages

### Connection (`connection.py`)
- **DatabaseManager**: Handles database connections, session management, and table creation

### Services (`services/`)
- **PDFDocumentService**: Operations for PDF documents (create, retrieve by folder, list folders)
- **PDFChunkService**: Operations for PDF chunks (create chunks, similarity search)
- **ChatService**: Operations for chat sessions and messages

## Usage

### Basic Setup
```python
from database import DatabaseManager, PDFDocumentService, PDFChunkService, ChatService

# Initialize database manager
db_manager = DatabaseManager("postgresql://user:pass@localhost/dbname")
db_manager.create_tables()

# Initialize services
pdf_doc_service = PDFDocumentService(db_manager)
pdf_chunk_service = PDFChunkService(db_manager)
chat_service = ChatService(db_manager)
```

### PDF Operations
```python
# Create a document
document = pdf_doc_service.create_document("folder1", "file.pdf", pdf_bytes)

# Get documents by folder
documents = pdf_doc_service.get_documents_by_folder("folder1")

# Create chunks with embeddings
chunks_data = [
    {"index": 0, "text": "chunk text", "embedding": [0.1, 0.2, ...]},
    # ... more chunks
]
pdf_chunk_service.create_chunks(document.id, chunks_data)

# Search similar chunks
results = pdf_chunk_service.search_similar_chunks(query_embedding, "folder1", limit=5)
```

### Chat Operations
```python
# Create a chat session
session = chat_service.create_session("folder1", "gpt-4")

# Add messages
chat_service.add_message(session.id, "user", "Hello")
chat_service.add_message(session.id, "assistant", "Hi there!")

# Get session messages
messages = chat_service.get_session_messages(session.id)
```

## Migration from Old Structure

The old monolithic `db.py` file has been refactored into this modular structure. The main `db.py` file now imports from this module for backward compatibility, so existing code should continue to work without changes.

## Benefits of Modular Structure

1. **Separation of Concerns**: Each file has a specific responsibility
2. **Maintainability**: Easier to find and modify specific functionality
3. **Testability**: Individual components can be tested in isolation
4. **Scalability**: Easy to add new models or services
5. **Code Organization**: Logical grouping of related functionality 