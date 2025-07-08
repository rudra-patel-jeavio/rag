#!/usr/bin/env python3
"""
Test script to verify embedding configuration is working correctly.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_embedding_config():
    """Test the embedding configuration."""
    print("🔍 Testing embedding configuration...")
    
    # Check OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    print(f"OpenAI API Key: {openai_key}")
    
    if openai_key and openai_key != "your_openai_api_key_here" and openai_key.strip():
        print("✅ OpenAI API key detected")
        has_openai = True
    else:
        print("❌ No valid OpenAI API key found")
        has_openai = False
    
    # Test embedding model selection
    from embedding import get_recommended_embedding_model
    
    try:
        model_type, model_name = get_recommended_embedding_model()
        print(f"📊 Recommended embedding model: {model_type}/{model_name}")
        
        # Test embedding generation
        from embedding import EmbeddingGenerator
        
        generator = EmbeddingGenerator(model_type=model_type, model_name=model_name)
        test_text = "This is a test sentence for embedding generation."
        
        print("🧪 Testing embedding generation...")
        embedding = generator.generate_embedding(test_text)
        
        print(f"✅ Successfully generated embedding with {len(embedding)} dimensions")
        print(f"📏 Embedding type: {type(embedding)}")
        print(f"🔢 First 5 values: {embedding[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing embeddings: {e}")
        return False

if __name__ == "__main__":
    success = test_embedding_config()
    if success:
        print("\n🎉 Embedding configuration test passed!")
    else:
        print("\n💥 Embedding configuration test failed!") 