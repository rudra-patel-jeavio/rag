#!/usr/bin/env python3
"""
Test script to verify vector search functionality after database migration.
"""

import os
import logging
from database.connection import DatabaseManager
from database.services.pdf_service import PDFChunkService
from embedding import QueryEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vector_search():
    """Test vector search functionality."""
    
    # Get database URL
    database_url = os.getenv('DATABASE_URL', 'postgresql://rag_user:rag_password@localhost:5433/rag_db')
    
    try:
        # Initialize services
        db_manager = DatabaseManager(database_url)
        chunk_service = PDFChunkService(db_manager)
        query_embedder = QueryEmbedder()
        
        # Test query
        test_query = "What is machine learning?"
        logger.info(f"Testing vector search with query: '{test_query}'")
        
        # Generate query embedding
        query_embedding = query_embedder.embed_query(test_query)
        logger.info(f"Generated query embedding with {len(query_embedding)} dimensions")
        
        # Perform similarity search
        results = chunk_service.search_similar_chunks(
            query_embedding=query_embedding,
            limit=3
        )
        
        logger.info(f"Found {len(results)} similar chunks:")
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. File: {result['file_name']}")
            logger.info(f"     Distance: {result['distance']:.4f}")
            logger.info(f"     Text preview: {result['text'][:100]}...")
            logger.info("")
        
        if results:
            logger.info("✅ Vector search is working correctly!")
        else:
            logger.warning("⚠️  No results found - this might be normal if no documents are uploaded")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Vector search test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_vector_search()
    if success:
        print("Vector search test completed successfully!")
    else:
        print("Vector search test failed!")
        exit(1) 