"""
Retrieval-Augmented Generation (RAG) functionality.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from db import DatabaseManager, PDFDocumentService, PDFChunkService, ChatService
from embedding import DocumentProcessor, QueryEmbedder, get_recommended_embedding_model
from llm_router import LLMRouter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """Main RAG system that orchestrates document processing, retrieval, and generation."""
    
    def __init__(self, 
                 db_manager: DatabaseManager,
                 embedding_model_type: str = "openai",
                 embedding_model_name: str = "text-embedding-ada-002",
                 chunk_size: int = 500,
                 overlap: int = 50):
        
        # Initialize services
        self.db_manager = db_manager
        self.pdf_service = PDFDocumentService(db_manager)
        self.chunk_service = PDFChunkService(db_manager)
        self.chat_service = ChatService(db_manager)
        
        # Initialize processors
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            overlap=overlap,
            embedding_model_type=embedding_model_type,
            embedding_model_name=embedding_model_name
        )
        
        self.query_embedder = QueryEmbedder(
            embedding_model_type=embedding_model_type,
            embedding_model_name=embedding_model_name
        )
        
        # Initialize LLM router
        self.llm_router = LLMRouter()
    
    def upload_and_process_pdf(self, 
                              pdf_data: bytes, 
                              file_name: str, 
                              folder_name: str) -> Dict[str, Any]:
        """Upload and process a PDF document."""
        try:
            logger.info(f"Processing PDF: {file_name} in folder: {folder_name}")
            
            # Create document record
            document = self.pdf_service.create_document(
                folder_name=folder_name,
                file_name=file_name,
                pdf_data=pdf_data
            )
            
            # Process PDF to extract text and generate embeddings
            processing_result = self.document_processor.process_pdf(pdf_data)
            
            # Store chunks with embeddings
            self.chunk_service.create_chunks(
                document_id=document.id,
                chunks_data=processing_result['chunks_data']
            )
            
            logger.info(f"Successfully processed {file_name}: {processing_result['total_chunks']} chunks created")
            
            return {
                'document_id': str(document.id),
                'file_name': file_name,
                'folder_name': folder_name,
                'total_chunks': processing_result['total_chunks'],
                'total_tokens': processing_result['total_tokens'],
                'success': True
            }
        
        except Exception as e:
            logger.error(f"Error processing PDF {file_name}: {e}")
            return {
                'file_name': file_name,
                'error': str(e),
                'success': False
            }
    
    def search_documents(self, 
                        query: str, 
                        folder_name: Optional[str] = None, 
                        top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant document chunks."""
        try:
            logger.info(f"Searching for query: '{query}' in folder: {folder_name}")
            
            # Generate query embedding
            query_embedding = self.query_embedder.embed_query(query)
            
            # Search for similar chunks
            results = self.chunk_service.search_similar_chunks(
                query_embedding=query_embedding,
                folder_name=folder_name,
                limit=top_k
            )
            
            logger.info(f"Found {len(results)} relevant chunks")
            return results
        
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise
    
    def generate_answer(self, 
                       query: str, 
                       model: str,
                       folder_name: Optional[str] = None, 
                       top_k: int = 5,
                       temperature: float = 0.7,
                       max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate answer using RAG."""
        try:
            # Search for relevant chunks
            relevant_chunks = self.search_documents(
                query=query,
                folder_name=folder_name,
                top_k=top_k
            )
            
            if not relevant_chunks:
                return {
                    'answer': "I couldn't find any relevant information in the documents to answer your question.",
                    'sources': [],
                    'query': query,
                    'model_used': model
                }
            
            # Generate response using LLM
            answer = self.llm_router.generate_rag_response(
                model=model,
                query=query,
                context_chunks=relevant_chunks,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Prepare sources information
            sources = self._prepare_sources(relevant_chunks)
            
            return {
                'answer': answer,
                'sources': sources,
                'query': query,
                'model_used': model,
                'folder_searched': folder_name or "All folders"
            }
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'query': query,
                'model_used': model,
                'error': str(e)
            }
    
    def _prepare_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare source information from chunks."""
        sources = []
        seen_files = set()
        
        for chunk in chunks:
            file_key = f"{chunk['folder_name']}/{chunk['file_name']}"
            if file_key not in seen_files:
                sources.append({
                    'file_name': chunk['file_name'],
                    'folder_name': chunk['folder_name'],
                    'relevance_score': 1 - chunk['distance']  # Convert distance to similarity
                })
                seen_files.add(file_key)
        
        return sources
    
    def get_folders(self) -> List[str]:
        """Get all available folders."""
        try:
            return self.pdf_service.get_all_folders()
        except Exception as e:
            logger.error(f"Error getting folders: {e}")
            return []
    
    def get_documents_in_folder(self, folder_name: str) -> List[Dict[str, Any]]:
        """Get all documents in a specific folder."""
        try:
            documents = self.pdf_service.get_documents_by_folder(folder_name)
            return [
                {
                    'id': str(doc['id']),
                    'file_name': doc['file_name'],
                    'uploaded_at': doc['uploaded_at'].isoformat(),
                    'chunk_count': doc['chunk_count']
                }
                for doc in documents
            ]
        except Exception as e:
            logger.error(f"Error getting documents in folder {folder_name}: {e}")
            return []

class ChatRAG:
    """Chat interface for RAG system with conversation history."""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.current_session = None
        self.conversation_history = []
    
    def start_chat_session(self, folder_name: Optional[str], model: str):
        """Start a new chat session."""
        try:
            self.current_session = self.rag_system.chat_service.create_session(
                folder_name=folder_name,
                model_used=model
            )
            self.conversation_history = []
            logger.info(f"Started chat session {self.current_session.id}")
        except Exception as e:
            logger.error(f"Error starting chat session: {e}")
            raise
    
    def chat(self, 
             query: str, 
             model: str,
             folder_name: Optional[str] = None,
             top_k: int = 10,
             temperature: float = 0.2,
             max_tokens: int = 1000) -> Dict[str, Any]:
        """Process a chat message with RAG."""
        try:
            # Generate RAG response
            response = self.rag_system.generate_answer(
                query=query,
                model=model,
                folder_name=folder_name,
                top_k=top_k,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Add to conversation history
            self.conversation_history.append({
                'role': 'user',
                'content': query,
                'timestamp': None
            })
            
            self.conversation_history.append({
                'role': 'assistant',
                'content': response['answer'],
                'timestamp': None,
                'sources': response.get('sources', [])
            })
            
            # Store in database if session exists
            if self.current_session:
                self.rag_system.chat_service.add_message(
                    session_id=self.current_session.id,
                    role='user',
                    message=query
                )
                
                self.rag_system.chat_service.add_message(
                    session_id=self.current_session.id,
                    role='assistant',
                    message=response['answer']
                )
            
            return response
        
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return {
                'answer': f"I encountered an error: {str(e)}",
                'sources': [],
                'query': query,
                'model_used': model,
                'error': str(e)
            }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get current conversation history."""
        return self.conversation_history
    
    def clear_conversation(self):
        """Clear current conversation history."""
        self.conversation_history = []

class DocumentManager:
    """Manages document operations and statistics."""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            folders = self.rag_system.get_folders()
            total_documents = 0
            folder_stats = []
            
            for folder in folders:
                documents = self.rag_system.get_documents_in_folder(folder)
                doc_count = len(documents)
                total_documents += doc_count
                
                folder_stats.append({
                    'folder_name': folder,
                    'document_count': doc_count,
                    'documents': documents
                })
            
            return {
                'total_folders': len(folders),
                'total_documents': total_documents,
                'folder_stats': folder_stats
            }
        
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {
                'total_folders': 0,
                'total_documents': 0,
                'folder_stats': [],
                'error': str(e)
            }
    
    def search_preview(self, 
                      query: str, 
                      folder_name: Optional[str] = None, 
                      top_k: int = 3) -> List[Dict[str, Any]]:
        """Get a preview of search results without generating full answer."""
        try:
            results = self.rag_system.search_documents(
                query=query,
                folder_name=folder_name,
                top_k=top_k
            )
            
            # Format for preview
            preview_results = []
            for result in results:
                preview_results.append({
                    'file_name': result['file_name'],
                    'folder_name': result['folder_name'],
                    'text_preview': result['text'][:200] + "..." if len(result['text']) > 200 else result['text'],
                    'relevance_score': 1 - result['distance']
                })
            
            return preview_results
        
        except Exception as e:
            logger.error(f"Error in search preview: {e}")
            return []

# Utility functions
def initialize_rag_system(database_url: str, 
                         embedding_model_type: str = "openai",
                         embedding_model_name: str = "text-embedding-ada-002") -> RAGSystem:
    """Initialize the RAG system with database and embedding configuration."""
    try:
        # Auto-detect best embedding model if using defaults
        if embedding_model_type == "openai" and embedding_model_name == "text-embedding-ada-002":
            recommended_type, recommended_name = get_recommended_embedding_model()
            embedding_model_type = recommended_type
            embedding_model_name = recommended_name
            logger.info(f"Using recommended embedding model: {embedding_model_type}/{embedding_model_name}")
        
        # Initialize database manager
        db_manager = DatabaseManager(database_url)
        
        # Create database tables if they don't exist
        logger.info("Creating database tables...")
        db_manager.create_tables()
        logger.info("Database tables created successfully")
        
        # Initialize RAG system
        rag_system = RAGSystem(
            db_manager=db_manager,
            embedding_model_type=embedding_model_type,
            embedding_model_name=embedding_model_name
        )
        
        logger.info("RAG system initialized successfully")
        return rag_system
    
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise 