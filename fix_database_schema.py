#!/usr/bin/env python3
"""
Database schema migration script to fix embedding column type.
This script converts the embedding column from TEXT to VECTOR type for pgvector compatibility.
"""

import os
import json
import logging
from sqlalchemy import create_engine, text, MetaData, Table, Column
from sqlalchemy.orm import sessionmaker
from database.connection import DatabaseManager
from database.models import Base, PDFChunk, PDFDocument

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_database(database_url: str):
    """Migrate database schema to fix embedding column type."""
    
    logger.info("Starting database migration...")
    
    # Create engine and session
    engine = create_engine(database_url, echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    with engine.connect() as conn:
        # Start transaction
        trans = conn.begin()
        
        try:
            # Step 1: Check if we need to migrate
            logger.info("Checking current schema...")
            result = conn.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'pdf_chunks' AND column_name = 'embedding'
            """))
            
            current_schema = result.fetchone()
            if current_schema and current_schema[1] == 'vector':
                logger.info("Database schema is already correct. No migration needed.")
                trans.rollback()
                return
            
            logger.info(f"Current embedding column type: {current_schema[1] if current_schema else 'Not found'}")
            
            # Step 2: Create backup table
            logger.info("Creating backup table...")
            conn.execute(text("""
                CREATE TABLE pdf_chunks_backup AS 
                SELECT * FROM pdf_chunks
            """))
            
            # Step 3: Drop the existing table
            logger.info("Dropping existing pdf_chunks table...")
            conn.execute(text("DROP TABLE pdf_chunks CASCADE"))
            
            # Step 4: Recreate table with proper schema
            logger.info("Recreating table with proper vector type...")
            
            # Commit the transaction so far
            trans.commit()
            
            # Create new transaction for table creation
            trans = conn.begin()
            
            # Create the table with proper schema using SQLAlchemy
            Base.metadata.create_all(bind=engine, tables=[PDFChunk.__table__])
            
            # Step 5: Migrate data back
            logger.info("Migrating data back to new table...")
            
            # Get all backup data
            result = conn.execute(text("""
                SELECT id, document_id, chunk_index, chunk_text, embedding, created_at
                FROM pdf_chunks_backup
            """))
            
            backup_data = result.fetchall()
            logger.info(f"Found {len(backup_data)} chunks to migrate")
            
            # Insert data back with proper vector format
            for row in backup_data:
                try:
                    # Parse the JSON embedding
                    if isinstance(row[4], str):
                        embedding_list = json.loads(row[4])
                    else:
                        embedding_list = row[4]
                    
                    # Insert with proper vector format
                    conn.execute(text("""
                        INSERT INTO pdf_chunks (id, document_id, chunk_index, chunk_text, embedding, created_at)
                        VALUES (:id, :document_id, :chunk_index, :chunk_text, :embedding, :created_at)
                    """), {
                        'id': row[0],
                        'document_id': row[1], 
                        'chunk_index': row[2],
                        'chunk_text': row[3],
                        'embedding': embedding_list,
                        'created_at': row[5]
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to migrate chunk {row[0]}: {e}")
                    continue
            
            # Step 6: Recreate foreign key constraints
            logger.info("Recreating foreign key constraints...")
            conn.execute(text("""
                ALTER TABLE pdf_chunks 
                ADD CONSTRAINT pdf_chunks_document_id_fkey 
                FOREIGN KEY (document_id) REFERENCES pdf_documents(id)
            """))
            
            # Step 7: Create indexes for better performance
            logger.info("Creating performance indexes...")
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_pdf_chunks_document_id 
                ON pdf_chunks(document_id)
            """))
            
            # Create vector index (HNSW is better for most use cases than IVFFlat)
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_pdf_chunks_embedding_hnsw 
                ON pdf_chunks USING hnsw (embedding vector_cosine_ops)
            """))
            
            # Step 8: Clean up backup table
            logger.info("Cleaning up backup table...")
            conn.execute(text("DROP TABLE pdf_chunks_backup"))
            
            # Commit all changes
            trans.commit()
            logger.info("Database migration completed successfully!")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            trans.rollback()
            
            # Try to restore from backup if it exists
            try:
                logger.info("Attempting to restore from backup...")
                conn.execute(text("DROP TABLE IF EXISTS pdf_chunks"))
                conn.execute(text("ALTER TABLE pdf_chunks_backup RENAME TO pdf_chunks"))
                logger.info("Restored from backup successfully")
            except Exception as restore_error:
                logger.error(f"Failed to restore from backup: {restore_error}")
            
            raise

def main():
    """Main function to run the migration."""
    
    # Get database URL from environment or use default
    database_url = os.getenv('DATABASE_URL', 'postgresql://rag_user:rag_password@localhost:5433/rag_db')
    
    if not database_url.startswith('postgresql'):
        logger.info("Migration only needed for PostgreSQL databases. Skipping.")
        return
    
    try:
        migrate_database(database_url)
        logger.info("Migration completed successfully!")
        
        # Test the migration
        logger.info("Testing migrated database...")
        db_manager = DatabaseManager(database_url)
        session = db_manager.get_session()
        
        try:
            # Try a simple query to verify the schema works
            chunk_count = session.query(PDFChunk).count()
            logger.info(f"Successfully queried {chunk_count} chunks from migrated database")
        finally:
            db_manager.close_session(session)
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    main() 