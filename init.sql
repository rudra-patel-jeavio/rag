-- Initialize the database with pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create indexes for better performance
-- Note: These will be created by SQLAlchemy, but we can add them here for optimization

-- Index for folder-based queries
-- CREATE INDEX IF NOT EXISTS idx_pdf_documents_folder ON pdf_documents(folder_name);

-- Vector similarity index (will be created after data is inserted)
-- CREATE INDEX IF NOT EXISTS idx_pdf_chunks_embedding ON pdf_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Index for document relationships
-- CREATE INDEX IF NOT EXISTS idx_pdf_chunks_document_id ON pdf_chunks(document_id);

-- Index for chat sessions
-- CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
-- CREATE INDEX IF NOT EXISTS idx_chat_messages_timestamp ON chat_messages(timestamp);

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE rag_db TO rag_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO rag_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO rag_user;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO rag_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO rag_user; 