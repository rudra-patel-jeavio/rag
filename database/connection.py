"""
Database connection and session management for the RAG application.
"""

import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from .models import Base

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all tables and enable pgvector extension if using PostgreSQL."""
        try:
            # Only try to enable pgvector extension for PostgreSQL
            if self.database_url.startswith('postgresql'):
                try:
                    with self.engine.connect() as conn:
                        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                        conn.commit()
                    logger.info("PostgreSQL pgvector extension enabled")
                except Exception as e:
                    logger.warning(f"Could not enable pgvector extension: {e}")
                    logger.info("Vector similarity search will fall back to Python calculations")
            else:
                logger.info("Using SQLite database - no extensions needed")
            
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