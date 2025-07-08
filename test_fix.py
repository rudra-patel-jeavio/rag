#!/usr/bin/env python3
"""
Test script to verify the database initialization fix
"""

import os
from dotenv import load_dotenv
from database.connection import DatabaseManager

def test_database_initialization():
    """Test that database initialization works with the text() fix."""
    load_dotenv()
    
    database_url = os.getenv('DATABASE_URL')
    print(f"Testing database initialization with: {database_url}")
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager(database_url)
        
        # This should now work without the "Not an executable object" error
        db_manager.create_tables()
        
        print("✅ Database initialization successful!")
        print("✅ pgvector extension enabled successfully!")
        print("✅ Tables created successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

if __name__ == "__main__":
    success = test_database_initialization()
    if success:
        print("\n🎉 Fix verified! The RAG system should now work properly.")
    else:
        print("\n❌ Fix failed. Please check the error above.") 