"""
Text chunking and embedding generation for the RAG application.
"""

import os
import re
import tiktoken
from typing import List, Dict, Any
import fitz  # PyMuPDF
import openai
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy import for sentence_transformers to avoid torch issues during startup
_sentence_transformer_models = {}

def _get_sentence_transformer_model(model_name: str):
    """Lazy loader for SentenceTransformer to avoid torch import issues."""
    global _sentence_transformer_models
    if model_name not in _sentence_transformer_models:
        try:
            # Import sentence_transformers only when needed
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence transformer model: {model_name}")
            _sentence_transformer_models[model_name] = SentenceTransformer(model_name)
        except ImportError as e:
            logger.error(f"Failed to import sentence_transformers: {e}")
            raise ImportError("sentence_transformers is required but not installed. Please install it with: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            raise
    return _sentence_transformer_models[model_name]

class TextChunker:
    """Handles text chunking with token-based splitting."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50, encoding_name: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def split_text_by_tokens(self, text: str) -> List[str]:
        """Split text into chunks based on token count."""
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_tokens = self.count_tokens(current_chunk)
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/]', '', text)
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be improved with spaCy or NLTK
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= self.overlap:
            return text
        
        overlap_tokens = tokens[-self.overlap:]
        return self.encoding.decode(overlap_tokens)

class PDFProcessor:
    """Handles PDF text extraction."""
    
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, pdf_data: bytes) -> str:
        """Extract text from PDF binary data using PyMuPDF."""
        try:
            # Open PDF from bytes
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            
            text_content = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text_content += page.get_text()
                text_content += "\n\n"  # Add page separator
            
            pdf_document.close()
            return text_content.strip()
        
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

class EmbeddingGenerator:
    """Handles embedding generation using different models."""
    
    def __init__(self, model_type: str = "openai", model_name: str = "text-embedding-ada-002"):
        self.model_type = model_type
        self.model_name = model_name
        
        # Auto-detect best available embedding model if OpenAI key is not available
        if model_type == "openai" and not os.getenv("OPENAI_API_KEY"):
            logger.warning("OpenAI API key not found, falling back to sentence_transformers")
            self.model_type = "sentence_transformers"
            self.model_name = "all-MiniLM-L6-v2"  # Fast and efficient local model
        
        if self.model_type == "openai":
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.model_type == "sentence_transformers":
            self.model = _get_sentence_transformer_model(self.model_name)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            if self.model_type == "openai":
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=text
                )
                return response.data[0].embedding
            
            elif self.model_type == "sentence_transformers":
                embedding = self.model.encode(text)
                return embedding.tolist()
        
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            if self.model_type == "openai":
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=texts
                )
                return [data.embedding for data in response.data]
            
            elif self.model_type == "sentence_transformers":
                embeddings = self.model.encode(texts)
                return embeddings.tolist()
        
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise

class DocumentProcessor:
    """Main class for processing documents end-to-end."""
    
    def __init__(self, 
                 chunk_size: int = 600, 
                 overlap: int = 50,
                 embedding_model_type: str = "openai",
                 embedding_model_name: str = "text-embedding-ada-002"):
        
        # Auto-detect best available embedding model if OpenAI key is not available
        if embedding_model_type == "openai" and not os.getenv("OPENAI_API_KEY"):
            logger.warning("OpenAI API key not found, using sentence_transformers for embeddings")
            embedding_model_type = "sentence_transformers"
            embedding_model_name = "all-MiniLM-L6-v2"
        
        self.pdf_processor = PDFProcessor()
        self.text_chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedding_generator = EmbeddingGenerator(
            model_type=embedding_model_type,
            model_name=embedding_model_name
        )
    
    def process_pdf(self, pdf_data: bytes) -> Dict[str, Any]:
        """Process PDF from upload to chunks with embeddings."""
        try:
            # Extract text from PDF
            logger.info("Extracting text from PDF...")
            text_content = self.pdf_processor.extract_text_from_pdf(pdf_data)
            
            if not text_content.strip():
                raise ValueError("No text content found in PDF")
            
            # Split text into chunks
            logger.info("Splitting text into chunks...")
            chunks = self.text_chunker.split_text_by_tokens(text_content)
            
            if not chunks:
                raise ValueError("No chunks generated from text")
            
            # Generate embeddings for chunks
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings = self.embedding_generator.generate_embeddings_batch(chunks)
            
            # Prepare chunk data
            chunks_data = []
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                chunks_data.append({
                    'index': i,
                    'text': chunk_text,
                    'embedding': embedding,
                    'token_count': self.text_chunker.count_tokens(chunk_text)
                })
            
            return {
                'full_text': text_content,
                'chunks_data': chunks_data,
                'total_chunks': len(chunks_data),
                'total_tokens': sum(chunk['token_count'] for chunk in chunks_data)
            }
        
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise

class QueryEmbedder:
    """Handles query embedding for similarity search."""
    
    def __init__(self, 
                 embedding_model_type: str = "openai",
                 embedding_model_name: str = "text-embedding-ada-002"):
        
        # Auto-detect best available embedding model if OpenAI key is not available
        if embedding_model_type == "openai" and not os.getenv("OPENAI_API_KEY"):
            logger.warning("OpenAI API key not found, using sentence_transformers for query embeddings")
            embedding_model_type = "sentence_transformers"
            embedding_model_name = "all-MiniLM-L6-v2"
        
        self.embedding_generator = EmbeddingGenerator(
            model_type=embedding_model_type,
            model_name=embedding_model_name
        )
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for search query."""
        try:
            return self.embedding_generator.generate_embedding(query)
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise

# Utility functions
def get_available_embedding_models() -> Dict[str, List[str]]:
    """Get available embedding models."""
    return {
        "openai": [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large"
        ],
        "sentence_transformers": [
            "all-MiniLM-L6-v2",  # Fast, lightweight, good for most use cases
            "all-mpnet-base-v2",  # Better quality, slower
            "multi-qa-MiniLM-L6-cos-v1",  # Optimized for Q&A
            "paraphrase-MiniLM-L6-v2",  # Good for semantic similarity
            "all-distilroberta-v1",  # Balanced speed and quality
            "sentence-transformers/all-MiniLM-L12-v2"  # Larger, better quality
        ]
    }

def get_recommended_embedding_model() -> tuple:
    """Get recommended embedding model based on available API keys."""
    if os.getenv("OPENAI_API_KEY"):
        return ("openai", "text-embedding-ada-002")
    else:
        logger.info("No OpenAI API key found, recommending local sentence transformer model")
        return ("sentence_transformers", "all-MiniLM-L6-v2")

def validate_pdf_file(file_data: bytes) -> bool:
    """Validate if the file is a valid PDF."""
    try:
        pdf_document = fitz.open(stream=file_data, filetype="pdf")
        page_count = pdf_document.page_count
        pdf_document.close()
        return page_count > 0
    except:
        return False 