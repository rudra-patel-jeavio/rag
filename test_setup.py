#!/usr/bin/env python3
"""
Test script to verify the RAG system setup and dependencies.
"""

import os
import sys
import logging
from typing import Dict, Any

def test_imports() -> Dict[str, bool]:
    """Test if all required modules can be imported."""
    results = {}
    
    # Core dependencies
    modules = [
        'streamlit',
        'sqlalchemy',
        'psycopg2',
        'pgvector',
        'openai',
        'litellm',
        'fitz',  # PyMuPDF
        'sentence_transformers',
        'numpy',
        'pandas',
        'tiktoken'
    ]
    
    for module in modules:
        try:
            __import__(module)
            results[module] = True
            print(f"✅ {module}")
        except ImportError as e:
            results[module] = False
            print(f"❌ {module}: {e}")
    
    return results

def test_environment_variables() -> Dict[str, bool]:
    """Test if required environment variables are set."""
    results = {}
    
    # Required variables
    required_vars = ['DATABASE_URL']
    
    # Optional but recommended variables
    optional_vars = [
        'OPENAI_API_KEY',
        'ANTHROPIC_API_KEY',
        'COHERE_API_KEY',
        'MISTRAL_API_KEY',
        'GROQ_API_KEY'
    ]
    
    print("\n🔧 Environment Variables:")
    
    for var in required_vars:
        value = os.getenv(var)
        results[var] = bool(value)
        status = "✅" if value else "❌"
        print(f"{status} {var}: {'Set' if value else 'Not set'}")
    
    # Check if at least one LLM API key is set
    llm_keys_set = any(os.getenv(var) for var in optional_vars)
    results['llm_api_keys'] = llm_keys_set
    status = "✅" if llm_keys_set else "⚠️"
    print(f"{status} LLM API Keys: {'At least one set' if llm_keys_set else 'None set'}")
    
    for var in optional_vars:
        value = os.getenv(var)
        status = "✅" if value else "⚠️"
        print(f"{status} {var}: {'Set' if value else 'Not set'}")
    
    return results

def test_database_connection() -> bool:
    """Test database connection."""
    try:
        from sqlalchemy import create_engine, text
        
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            print("❌ DATABASE_URL not set")
            return False
        
        print(f"\n🗄️ Testing database connection...")
        engine = create_engine(database_url)
        
        with engine.connect() as conn:
            # Test basic connection
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
            
            # Test pgvector extension
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                print("✅ Database connection successful")
                print("✅ pgvector extension available")
                return True
            except Exception as e:
                print(f"⚠️ pgvector extension issue: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_rag_system_initialization() -> bool:
    """Test RAG system initialization."""
    try:
        print(f"\n🤖 Testing RAG system initialization...")
        
        # Import our modules
        from rag import initialize_rag_system
        
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            print("❌ Cannot test RAG system: DATABASE_URL not set")
            return False
        
        # Initialize RAG system
        rag_system = initialize_rag_system(database_url)
        
        # Test basic functionality
        folders = rag_system.get_folders()
        print(f"✅ RAG system initialized successfully")
        print(f"✅ Found {len(folders)} folders in database")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG system initialization failed: {e}")
        return False

def test_llm_availability() -> Dict[str, bool]:
    """Test LLM model availability."""
    results = {}
    
    try:
        print(f"\n🧠 Testing LLM availability...")
        
        from llm_router import LLMRouter, validate_api_keys
        
        # Check API keys
        api_status = validate_api_keys()
        print("API Key Status:")
        for provider, available in api_status.items():
            status = "✅" if available else "❌"
            print(f"  {status} {provider}")
        
        # Test LLM router
        if any(api_status.values()):
            llm_router = LLMRouter()
            available_models = llm_router.get_model_list_for_ui()
            print(f"✅ LLM Router initialized with {len(available_models)} models")
            results['llm_router'] = True
        else:
            print("⚠️ No API keys available for LLM testing")
            results['llm_router'] = False
        
        return results
        
    except Exception as e:
        print(f"❌ LLM testing failed: {e}")
        return {'llm_router': False}

def main():
    """Run all tests."""
    print("🧪 RAG System Setup Test\n")
    print("=" * 50)
    
    # Test imports
    print("📦 Testing imports...")
    import_results = test_imports()
    
    # Test environment variables
    env_results = test_environment_variables()
    
    # Test database connection
    db_success = test_database_connection()
    
    # Test RAG system initialization
    rag_success = test_rag_system_initialization() if db_success else False
    
    # Test LLM availability
    llm_results = test_llm_availability()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    failed_imports = [k for k, v in import_results.items() if not v]
    if failed_imports:
        print(f"❌ Failed imports: {', '.join(failed_imports)}")
    else:
        print("✅ All imports successful")
    
    if not env_results.get('DATABASE_URL', False):
        print("❌ DATABASE_URL not configured")
    else:
        print("✅ DATABASE_URL configured")
    
    if not env_results.get('llm_api_keys', False):
        print("⚠️ No LLM API keys configured")
    else:
        print("✅ LLM API keys configured")
    
    if db_success:
        print("✅ Database connection working")
    else:
        print("❌ Database connection failed")
    
    if rag_success:
        print("✅ RAG system initialization working")
    else:
        print("❌ RAG system initialization failed")
    
    if llm_results.get('llm_router', False):
        print("✅ LLM router working")
    else:
        print("❌ LLM router failed")
    
    # Overall status
    all_critical_passed = (
        all(import_results.values()) and
        env_results.get('DATABASE_URL', False) and
        db_success and
        rag_success
    )
    
    print("\n" + "=" * 50)
    if all_critical_passed:
        print("🎉 All critical tests passed! Your RAG system is ready to use.")
        print("Run: streamlit run main.py")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        print("Refer to the README.md for setup instructions.")
    
    return 0 if all_critical_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 