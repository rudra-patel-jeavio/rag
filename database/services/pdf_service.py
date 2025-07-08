"""
PDF document and chunk services for the RAG application.
"""

import uuid
import json
import logging
import numpy as np
from typing import List, Optional
from ..models import PDFDocument, PDFChunk
from ..connection import DatabaseManager
from sqlalchemy import func

logger = logging.getLogger(__name__)

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def cosine_distance(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine distance (1 - cosine similarity) between two vectors."""
    return 1.0 - cosine_similarity(vec1, vec2)

class PDFDocumentService:
    """Service for PDF document operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def create_document(self, folder_name: str, file_name: str, pdf_data: bytes) -> PDFDocument:
        """Create a new PDF document record."""
        session = self.db_manager.get_session()
        try:
            document = PDFDocument(
                folder_name=folder_name,
                file_name=file_name,
                pdf_file=pdf_data
            )
            session.add(document)
            session.commit()
            session.refresh(document)
            return document
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating document: {e}")
            raise
        finally:
            self.db_manager.close_session(session)
    
    def get_documents_by_folder(self, folder_name: str) -> List[dict]:
        """Get all documents in a folder with chunk counts."""
        session = self.db_manager.get_session()
        try:
            # Query documents with chunk counts using a subquery
            # Create subquery to count chunks per document
            chunk_count_subquery = session.query(
                PDFChunk.document_id,
                func.count(PDFChunk.id).label('chunk_count')
            ).group_by(PDFChunk.document_id).subquery()
            
            # Query documents with their chunk counts
            results = session.query(
                PDFDocument.id,
                PDFDocument.folder_name,
                PDFDocument.file_name,
                PDFDocument.uploaded_at,
                func.coalesce(chunk_count_subquery.c.chunk_count, 0).label('chunk_count')
            ).outerjoin(
                chunk_count_subquery,
                PDFDocument.id == chunk_count_subquery.c.document_id
            ).filter(
                PDFDocument.folder_name == folder_name
            ).all()
            
            # Convert to dictionaries to avoid session dependency
            documents = []
            for result in results:
                documents.append({
                    'id': result.id,
                    'folder_name': result.folder_name,
                    'file_name': result.file_name,
                    'uploaded_at': result.uploaded_at,
                    'chunk_count': result.chunk_count or 0
                })
            
            return documents
        finally:
            self.db_manager.close_session(session)
    
    def get_all_folders(self) -> List[str]:
        """Get all unique folder names."""
        session = self.db_manager.get_session()
        try:
            folders = session.query(PDFDocument.folder_name).distinct().all()
            return [folder[0] for folder in folders]
        finally:
            self.db_manager.close_session(session)

class PDFChunkService:
    """Service for PDF chunk operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def create_chunks(self, document_id: uuid.UUID, chunks_data: List[dict]):
        """Create multiple chunks for a document."""
        session = self.db_manager.get_session()
        try:
            chunks = []
            for chunk_data in chunks_data:
                chunk = PDFChunk(
                    document_id=document_id,
                    chunk_index=chunk_data['index'],
                    chunk_text=chunk_data['text'],
                    embedding=chunk_data['embedding']
                )
                chunks.append(chunk)
            
            session.add_all(chunks)
            session.commit()
            return chunks
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating chunks: {e}")
            raise
        finally:
            self.db_manager.close_session(session)
    
    def search_similar_chunks(self, query_embedding: List[float], folder_name: Optional[str] = None, limit: int = 5) -> List[dict]:
        """Search for similar chunks using cosine similarity."""
        session = self.db_manager.get_session()
        try:
            # Check if we're using PostgreSQL with pgvector
            if self.db_manager.database_url.startswith('postgresql'):
                try:
                    # Try PostgreSQL with pgvector - ensure query_embedding is a list
                    if not isinstance(query_embedding, list):
                        query_embedding = list(query_embedding)
                    
                    query = session.query(
                        PDFChunk.chunk_text,
                        PDFDocument.file_name,
                        PDFDocument.folder_name,
                        PDFChunk.embedding.cosine_distance(query_embedding).label('distance')
                    ).join(PDFDocument)
                    
                    if folder_name:
                        query = query.filter(PDFDocument.folder_name == folder_name)
                    
                    results = query.order_by('distance').limit(limit).all()
                    
                    return [
                        {
                            'text': result.chunk_text,
                            'file_name': result.file_name,
                            'folder_name': result.folder_name,
                            'distance': float(result.distance)  # Ensure distance is a float
                        }
                        for result in results
                    ]
                except Exception as e:
                    logger.warning(f"PostgreSQL vector search failed, falling back to Python calculation: {e}")
                    # Rollback the failed transaction to clean up the session state
                    session.rollback()
            
            # Fallback for SQLite or if PostgreSQL vector search fails
            # Get all chunks (with folder filter if specified)
            query = session.query(
                PDFChunk.chunk_text,
                PDFChunk.embedding,
                PDFDocument.file_name,
                PDFDocument.folder_name
            ).join(PDFDocument)
            
            if folder_name:
                query = query.filter(PDFDocument.folder_name == folder_name)
            
            all_chunks = query.all()
            
            # Calculate similarities in Python
            chunk_similarities = []
            for chunk in all_chunks:
                try:
                    # Handle embedding conversion - it might be a string (JSON), list, or numpy array
                    chunk_embedding = chunk.embedding
                    
                    if isinstance(chunk_embedding, str):
                        try:
                            chunk_embedding = json.loads(chunk_embedding)
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(f"Failed to parse embedding for chunk, skipping")
                            continue
                    elif hasattr(chunk_embedding, 'tolist'):
                        chunk_embedding = chunk_embedding.tolist()
                    elif not isinstance(chunk_embedding, list):
                        logger.warning(f"Unexpected embedding type: {type(chunk_embedding)}, skipping")
                        continue
                    
                    # Ensure both embeddings are lists of floats
                    if not isinstance(query_embedding, list):
                        query_embedding = list(query_embedding)
                    
                    if not isinstance(chunk_embedding, list):
                        chunk_embedding = list(chunk_embedding)
                    
                    # Validate embedding dimensions
                    if len(query_embedding) != len(chunk_embedding):
                        logger.warning(f"Embedding dimension mismatch: query={len(query_embedding)}, chunk={len(chunk_embedding)}, skipping")
                        continue
                    
                    distance = cosine_distance(query_embedding, chunk_embedding)
                    chunk_similarities.append({
                        'text': chunk.chunk_text,
                        'file_name': chunk.file_name,
                        'folder_name': chunk.folder_name,
                        'distance': distance
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate distance for chunk: {e}")
                    continue
            
            # Sort by distance and return top results
            chunk_similarities.sort(key=lambda x: x['distance'])
            return chunk_similarities[:limit]
            
        finally:
            self.db_manager.close_session(session) 