"""
Main Streamlit application for the RAG system.
"""

import os
import warnings
import logging
import sys

# Comprehensive warning suppression for PyTorch/Streamlit compatibility
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*no running event loop.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Tried to instantiate class.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*torch.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.*")

# Suppress specific torch-related warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*__path__._path.*")
warnings.filterwarnings("ignore", message=".*_get_custom_class_python_wrapper.*")

# Configure logging to suppress noisy warnings before any torch imports
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Set environment variables to prevent torch issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Monkey patch to prevent Streamlit watcher issues with torch
def _safe_get_module_paths(module):
    """Safe version of get_module_paths that handles torch modules gracefully."""
    try:
        if hasattr(module, '__name__') and 'torch' in str(module.__name__):
            return []
        if hasattr(module, '__path__'):
            if hasattr(module.__path__, '_path'):
                return list(module.__path__._path)
            elif hasattr(module.__path__, '__iter__'):
                return list(module.__path__)
        return []
    except (RuntimeError, AttributeError, TypeError):
        return []

# Apply the monkey patch before importing streamlit
try:
    import streamlit.watcher.local_sources_watcher as watcher
    if hasattr(watcher, 'get_module_paths'):
        watcher.get_module_paths = _safe_get_module_paths
except ImportError:
    pass

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any

# Import our modules with error handling
try:
    from rag import initialize_rag_system, ChatRAG, DocumentManager
    from llm_router import validate_api_keys, get_default_model
    from embedding import validate_pdf_file
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG PDF Chat System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .source-box {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .stats-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    
    if 'chat_rag' not in st.session_state:
        st.session_state.chat_rag = None
    
    if 'document_manager' not in st.session_state:
        st.session_state.document_manager = None
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'current_folder' not in st.session_state:
        st.session_state.current_folder = None
    
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None

def setup_database_connection():
    """Setup database connection and initialize RAG system."""
    if st.session_state.rag_system is None:
        try:
            # Get database URL from environment or use SQLite as default
            database_url = os.getenv(
                "DATABASE_URL", 
                "sqlite:///./rag_database.db"
            )
            
            # Show database info
            if database_url.startswith("sqlite"):
                st.info("🗄️ Using SQLite database for local development")
            else:
                st.info(f"🗄️ Connecting to database: {database_url.split('@')[1] if '@' in database_url else 'configured database'}")
            
            # Initialize RAG system
            with st.spinner("Initializing RAG system..."):
                st.session_state.rag_system = initialize_rag_system(database_url)
                st.session_state.chat_rag = ChatRAG(st.session_state.rag_system)
                st.session_state.document_manager = DocumentManager(st.session_state.rag_system)
            
            st.success("✅ RAG system initialized successfully!")
            return True
        
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG system: {str(e)}")
            if "postgres" in str(e).lower() or "psycopg2" in str(e).lower():
                st.info("💡 **Tip**: For local development, you can use SQLite by setting:")
                st.code('DATABASE_URL="sqlite:///./rag_database.db"')
                st.info("Or install and configure PostgreSQL with the credentials from docker-compose.yml")
            else:
                st.info("Please check your database connection and API keys.")
            return False
    
    return True

def render_sidebar():
    """Render the sidebar with navigation and settings."""
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # API Keys Status
    st.sidebar.markdown("### 🔑 API Keys Status")
    api_status = validate_api_keys()
    for provider, available in api_status.items():
        status_icon = "✅" if available else "❌"
        st.sidebar.markdown(f"{status_icon} {provider}")
    
    # Model Selection
    st.sidebar.markdown("### 🤖 Model Selection")
    if st.session_state.rag_system:
        available_models = st.session_state.rag_system.llm_router.get_model_list_for_ui()
        if available_models:
            selected_model_display = st.sidebar.selectbox(
                "Choose LLM Model",
                available_models,
                index=0 if not st.session_state.current_model else 
                      (available_models.index(st.session_state.current_model) 
                       if st.session_state.current_model in available_models else 0)
            )
            st.session_state.current_model = selected_model_display
        else:
            st.sidebar.error("No models available. Check API keys.")
    
    # Folder Selection
    st.sidebar.markdown("### 📁 Folder Selection")
    if st.session_state.rag_system:
        folders = st.session_state.rag_system.get_folders()
        folder_options = ["All Folders"] + folders
        
        selected_folder = st.sidebar.selectbox(
            "Choose Folder",
            folder_options,
            index=0
        )
        
        st.session_state.current_folder = None if selected_folder == "All Folders" else selected_folder
    
    # Advanced Settings
    st.sidebar.markdown("### ⚙️ Advanced Settings")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    max_tokens = st.sidebar.slider("Max Tokens", 100, 2000, 1000, 100)
    top_k = st.sidebar.slider("Top-K Results", 1, 15, 10, 1)
    
    return {
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_k': top_k
    }

def render_upload_section():
    """Render the PDF upload section."""
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload PDF Documents")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to add to your knowledge base"
        )
    
    with col2:
        folder_name = st.text_input(
            "Folder Name",
            value="general",
            help="Enter a folder name to organize your documents"
        )
    
    if uploaded_files and folder_name:
        if st.button("🚀 Process Documents", type="primary"):
            process_uploaded_files(uploaded_files, folder_name)
    
    st.markdown('</div>', unsafe_allow_html=True)

def process_uploaded_files(uploaded_files, folder_name):
    """Process uploaded PDF files."""
    if not st.session_state.rag_system:
        st.error("RAG system not initialized!")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Read file data
            pdf_data = uploaded_file.read()
            
            # Validate PDF
            if not validate_pdf_file(pdf_data):
                st.error(f"❌ {uploaded_file.name} is not a valid PDF file")
                continue
            
            # Process PDF
            result = st.session_state.rag_system.upload_and_process_pdf(
                pdf_data=pdf_data,
                file_name=uploaded_file.name,
                folder_name=folder_name
            )
            
            results.append(result)
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
    
    # Display results
    status_text.empty()
    progress_bar.empty()
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    if successful:
        st.success(f"✅ Successfully processed {len(successful)} documents!")
        
        # Show processing stats
        total_chunks = sum(r['total_chunks'] for r in successful)
        total_tokens = sum(r['total_tokens'] for r in successful)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents Processed", len(successful))
        with col2:
            st.metric("Total Chunks", total_chunks)
        with col3:
            st.metric("Total Tokens", total_tokens)
    
    if failed:
        st.error(f"❌ Failed to process {len(failed)} documents")
        for failure in failed:
            st.error(f"• {failure['file_name']}: {failure.get('error', 'Unknown error')}")

def render_chat_interface(settings):
    """Render the chat interface."""
    st.markdown("### 💬 Chat with Your Documents")
    
    if not st.session_state.rag_system:
        st.warning("Please initialize the RAG system first.")
        return
    
    # Display current settings
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"📁 Folder: {st.session_state.current_folder or 'All Folders'}")
    with col2:
        if st.session_state.current_model:
            model_name = st.session_state.rag_system.llm_router.get_model_name_from_display(
                st.session_state.current_model
            )
            st.info(f"🤖 Model: {model_name}")
    with col3:
        st.info(f"🎯 Top-K: {settings['top_k']}")
    
    # Chat input
    user_query = st.chat_input("Ask a question about your documents...")
    
    if user_query:
        process_chat_query(user_query, settings)
    
    # Display conversation history
    display_conversation_history()
    
    # Clear conversation button
    if st.session_state.conversation_history:
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            if st.session_state.chat_rag:
                st.session_state.chat_rag.clear_conversation()
            st.rerun()

def process_chat_query(query: str, settings: Dict[str, Any]):
    """Process a chat query and generate response."""
    if not st.session_state.current_model:
        st.error("Please select a model first!")
        return
    
    model_name = st.session_state.rag_system.llm_router.get_model_name_from_display(
        st.session_state.current_model
    )
    
    with st.spinner("Generating response..."):
        try:
            response = st.session_state.chat_rag.chat(
                query=query,
                model=model_name,
                folder_name=st.session_state.current_folder,
                top_k=settings['top_k'],
                temperature=settings['temperature'],
                max_tokens=settings['max_tokens']
            )
            
            # Add to session state conversation history
            st.session_state.conversation_history.append({
                'role': 'user',
                'content': query,
                'timestamp': datetime.now()
            })
            
            st.session_state.conversation_history.append({
                'role': 'assistant',
                'content': response['answer'],
                'sources': response.get('sources', []),
                'timestamp': datetime.now()
            })
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

def display_conversation_history():
    """Display the conversation history."""
    for message in st.session_state.conversation_history:
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.write(message['content'])
        else:
            with st.chat_message("assistant"):
                st.write(message['content'])
                
                # Display sources if available
                if message.get('sources'):
                    with st.expander("📚 Sources"):
                        for source in message['sources']:
                            st.markdown(f"""
                            **{source['file_name']}** (Folder: {source['folder_name']})  
                            Relevance: {source['relevance_score']:.2f}
                            """)

def render_document_management():
    """Render document management interface."""
    st.markdown("### 📊 Document Management")
    
    if not st.session_state.document_manager:
        st.warning("Document manager not initialized.")
        return
    
    # Get system statistics
    stats = st.session_state.document_manager.get_system_stats()
    
    # Display overview stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Folders", stats['total_folders'])
    with col2:
        st.metric("Total Documents", stats['total_documents'])
    with col3:
        st.metric("System Status", "🟢 Active")
    
    # Display folder details
    if stats['folder_stats']:
        st.markdown("#### 📁 Folder Details")
        
        for folder_stat in stats['folder_stats']:
            with st.expander(f"📁 {folder_stat['folder_name']} ({folder_stat['document_count']} documents)"):
                if folder_stat['documents']:
                    df = pd.DataFrame(folder_stat['documents'])
                    df['uploaded_at'] = pd.to_datetime(df['uploaded_at']).dt.strftime('%Y-%m-%d %H:%M')
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No documents in this folder.")

def render_search_preview():
    """Render search preview functionality."""
    st.markdown("### 🔍 Search Preview")
    
    if not st.session_state.document_manager:
        st.warning("Document manager not initialized.")
        return
    
    search_query = st.text_input("Enter search query to preview results:")
    
    if search_query:
        folder_filter = st.selectbox(
            "Filter by folder:",
            ["All Folders"] + st.session_state.rag_system.get_folders()
        )
        
        folder_name = None if folder_filter == "All Folders" else folder_filter
        
        with st.spinner("Searching..."):
            try:
                results = st.session_state.document_manager.search_preview(
                    query=search_query,
                    folder_name=folder_name,
                    top_k=5
                )
                
                if results:
                    st.markdown(f"**Found {len(results)} relevant chunks:**")
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Result {i}: {result['file_name']} (Score: {result['relevance_score']:.2f})"):
                            st.markdown(f"**Folder:** {result['folder_name']}")
                            st.markdown(f"**Preview:** {result['text_preview']}")
                else:
                    st.info("No relevant results found.")
                    
            except Exception as e:
                st.error(f"Search error: {str(e)}")

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📚 RAG PDF Chat System</h1>', unsafe_allow_html=True)
    
    # Setup database connection
    if not setup_database_connection():
        st.stop()
    
    # Render sidebar
    settings = render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📤 Upload", "📊 Management", "🔍 Search"])
    
    with tab1:
        render_chat_interface(settings)
    
    with tab2:
        render_upload_section()
    
    with tab3:
        render_document_management()
    
    with tab4:
        render_search_preview()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit, PostgreSQL, pgvector, and LiteLLM | "
        "Upload PDFs, organize in folders, and chat with your documents!"
    )

if __name__ == "__main__":
    main() 