# RAG PDF Chat System

A comprehensive Retrieval-Augmented Generation (RAG) system built with Streamlit that allows you to upload PDF documents, organize them in folders, and chat with your documents using multiple LLM providers.

## 🚀 Features

### Core Functionality
- **PDF Upload & Processing**: Upload multiple PDFs with folder-based organization
- **Advanced Text Chunking**: Smart text splitting with ~500 tokens and 50 token overlap
- **Vector Embeddings**: Generate embeddings using OpenAI or HuggingFace models
- **Semantic Search**: Fast similarity search using PostgreSQL with pgvector
- **Multi-Model Chat**: Support for multiple LLM providers via LiteLLM

### Supported LLM Providers
- **OpenAI**: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, GPT-4o
- **Anthropic**: Claude 3 Opus, Sonnet, Haiku, Claude 3.5 Sonnet
- **Cohere**: Command R, Command R+
- **Mistral**: Large, Medium, Small models
- **Groq**: Llama 3 70B/8B, Mixtral 8x7B

### Key Features
- 📁 **Folder-based Organization**: Organize documents by topics/projects
- 🔍 **Smart Search**: Semantic search with relevance scoring
- 💬 **Interactive Chat**: Natural conversation with document context
- 📊 **Document Management**: View statistics and manage your knowledge base
- 🎛️ **Advanced Controls**: Temperature, token limits, top-k results
- 📚 **Source Citations**: See which documents informed each response
- 🔄 **Chat History**: Persistent conversation storage

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PostgreSQL 12+ with pgvector extension
- At least one LLM provider API key

### 1. Clone the Repository
```bash
git clone <repository-url>
cd project_rag
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Database Setup

#### Install PostgreSQL and pgvector
```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib
sudo apt-get install postgresql-14-pgvector

# macOS with Homebrew
brew install postgresql
brew install pgvector

# Or use Docker
docker run --name postgres-rag \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=rag_db \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
```

#### Create Database
```sql
CREATE DATABASE rag_db;
CREATE USER rag_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE rag_db TO rag_user;

-- Connect to rag_db and enable pgvector
\c rag_db
CREATE EXTENSION vector;
```

### 4. Environment Configuration
```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your configuration
nano .env
```

Required environment variables:
```bash
# Database
DATABASE_URL=postgresql://rag_user:your_password@localhost:5432/rag_db

# At least one LLM provider API key
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
COHERE_API_KEY=your_cohere_key
MISTRAL_API_KEY=your_mistral_key
GROQ_API_KEY=your_groq_key
```

### 5. Run the Application
```bash
streamlit run main.py
```

The application will be available at `http://localhost:8501`

## 📖 Usage Guide

### 1. Upload Documents
1. Navigate to the **Upload** tab
2. Select one or more PDF files
3. Enter a folder name (e.g., "research", "manuals", "reports")
4. Click "🚀 Process Documents"
5. Wait for processing to complete

### 2. Chat with Documents
1. Go to the **Chat** tab
2. Select your preferred LLM model from the sidebar
3. Choose a folder or "All Folders" for global search
4. Adjust advanced settings (temperature, max tokens, top-k)
5. Type your question and press Enter
6. View the response with source citations

### 3. Manage Documents
1. Visit the **Management** tab
2. View system statistics
3. Browse documents by folder
4. See processing details and chunk counts

### 4. Search Preview
1. Use the **Search** tab
2. Enter a query to preview relevant chunks
3. Filter by folder if needed
4. Review relevance scores and text previews

## 🏗️ Architecture

### Database Schema
```sql
-- PDF Documents
pdf_documents (
    id UUID PRIMARY KEY,
    folder_name TEXT,
    file_name TEXT,
    pdf_file BYTEA,
    uploaded_at TIMESTAMP
)

-- Text Chunks with Embeddings
pdf_chunks (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES pdf_documents(id),
    chunk_index INTEGER,
    chunk_text TEXT,
    embedding VECTOR(1536),
    created_at TIMESTAMP
)

-- Chat Sessions
chat_sessions (
    id UUID PRIMARY KEY,
    folder_name TEXT,
    model_used TEXT,
    started_at TIMESTAMP
)

-- Chat Messages
chat_messages (
    id UUID PRIMARY KEY,
    session_id UUID REFERENCES chat_sessions(id),
    role TEXT,
    message TEXT,
    timestamp TIMESTAMP
)
```

### Module Structure
- **`db.py`**: Database models and operations
- **`embedding.py`**: Text processing and embedding generation
- **`llm_router.py`**: Multi-model LLM integration via LiteLLM
- **`rag.py`**: Core RAG functionality and orchestration
- **`main.py`**: Streamlit UI and application logic

## ⚙️ Configuration

### Embedding Models
- **OpenAI**: `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`
- **HuggingFace**: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `multi-qa-MiniLM-L6-cos-v1`

### Chunking Parameters
- **Chunk Size**: ~500 tokens (configurable)
- **Overlap**: 50 tokens (configurable)
- **Encoding**: cl100k_base (GPT-4 tokenizer)

### Search Parameters
- **Top-K**: Number of relevant chunks to retrieve (1-10)
- **Similarity**: Cosine distance for vector search
- **Filtering**: Optional folder-based filtering

## 🐳 Docker Deployment (Optional)

### Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: rag_db
      POSTGRES_USER: rag_user
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  rag-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      DATABASE_URL: postgresql://rag_user:password@postgres:5432/rag_db
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    depends_on:
      - postgres
    volumes:
      - ./.env:/app/.env

volumes:
  postgres_data:
```

## 🔧 Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Verify PostgreSQL is running
   - Check DATABASE_URL format
   - Ensure pgvector extension is installed

2. **API Key Issues**
   - Verify API keys are correctly set in .env
   - Check API key permissions and quotas
   - Ensure at least one provider key is valid

3. **PDF Processing Errors**
   - Verify PDF files are not corrupted
   - Check file size limits
   - Ensure sufficient disk space

4. **Embedding Generation Fails**
   - Check API quotas and rate limits
   - Verify network connectivity
   - Consider switching to sentence-transformers for offline use

### Performance Optimization

1. **Database Indexing**
   ```sql
   CREATE INDEX idx_pdf_chunks_embedding ON pdf_chunks USING ivfflat (embedding vector_cosine_ops);
   CREATE INDEX idx_pdf_documents_folder ON pdf_documents(folder_name);
   ```

2. **Batch Processing**
   - Process multiple PDFs in smaller batches
   - Use sentence-transformers for faster local embeddings
   - Consider GPU acceleration for large documents

## 📝 API Reference

### Environment Variables
| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Yes | - |
| `OPENAI_API_KEY` | OpenAI API key | No* | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | No* | - |
| `COHERE_API_KEY` | Cohere API key | No* | - |
| `MISTRAL_API_KEY` | Mistral API key | No* | - |
| `GROQ_API_KEY` | Groq API key | No* | - |
| `EMBEDDING_MODEL_TYPE` | Embedding provider | No | openai |
| `EMBEDDING_MODEL_NAME` | Embedding model name | No | text-embedding-ada-002 |
| `CHUNK_SIZE` | Text chunk size in tokens | No | 500 |
| `CHUNK_OVERLAP` | Overlap between chunks | No | 50 |

*At least one LLM provider API key is required

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [LiteLLM](https://github.com/BerriAI/litellm) for unified LLM access
- [pgvector](https://github.com/pgvector/pgvector) for vector similarity search
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF text extraction
- [SQLAlchemy](https://sqlalchemy.org/) for database ORM

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing GitHub issues
3. Create a new issue with detailed information
4. Include logs and configuration details

---

**Built with ❤️ for the AI community** 