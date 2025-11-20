# RAG System - Retrieval-Augmented Generation

A scalable, production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI and Qdrant. This system enables semantic search and retrieval of relevant information from document collections, reducing LLM hallucinations and providing access to up-to-date, domain-specific knowledge.

## Overview

This RAG system addresses key limitations of Large Language Models by:

- **Reducing Hallucinations**: Grounding responses in actual document content
- **Enabling Knowledge Currency**: Accessing information not in the LLM's training data
- **Supporting Domain Expertise**: Leveraging proprietary documents and knowledge bases
- **Providing Scalability**: Efficiently searching large document collections

## Features

- PDF Document Processing: Extract and process text from PDF files
- Semantic Search: Vector-based similarity search using sentence transformers
- High Performance: FastAPI-based async API with sub-second query responses
- Vector Storage: Qdrant integration for efficient similarity search
- Comprehensive Logging: Structured logging with Sentry integration
- Well-Tested: Unit and integration tests with pytest
- Modular Design: Easy to extend with new document types or embedding models

## Architecture

```
┌─────────────┐
│   FastAPI   │  API Layer
│   Endpoints │
└──────┬──────┘
       │
┌──────▼──────────────────────────┐
│     Core Processing Layer       │
│  ┌──────────────────────────┐  │
│  │ Text Extraction          │  │
│  │ Chunking                 │  │
│  │ Embedding Generation     │  │
│  │ Vector Storage/Retrieval │  │
│  └──────────────────────────┘  │
└──────┬──────────────────────────┘
       │
┌──────▼──────┐
│   Qdrant    │  Vector Database
│  Vector DB  │
└─────────────┘
```

## Prerequisites

- Python 3.13+
- Qdrant vector database (local or cloud instance)
- uv package manager (recommended) or pip

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/abrarCSE29/Retrieval-Augmented-Generation.git
cd rag
```

### 2. Install Dependencies

Using uv (recommended):
```bash
uv sync
```

Using pip:
```bash
pip install -r requirements.txt
```

### 3. Set Up Qdrant

**Option A: Docker (Recommended)**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Option B: Qdrant Cloud**
Sign up at [cloud.qdrant.io](https://cloud.qdrant.io) and get your cluster URL and API key.

### 4. Configure Environment Variables

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` with your configuration:
```env
# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=  # Optional: path to log file
USE_JSON_LOGGING=false

# Sentry Error Tracking (Optional)
SENTRY_DSN=  # Your Sentry DSN if using error tracking
```

## Usage

### Starting the Server

```bash
# Using uv
uv run python main.py

# Or directly with Python
python main.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

#### 1. Health Check
```bash
GET /api/health
```

**Response:**
```json
{
  "status": "healthy"
}
```

#### 2. Upload Document
```bash
POST /api/documents
Content-Type: multipart/form-data
```

**Example with curl:**
```bash
curl -X POST "http://localhost:8000/api/documents" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "status": "success",
  "message": "Document processed and stored successfully",
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "chunks_count": 42
}
```

#### 3. Query Documents
```bash
POST /api/query
Content-Type: application/json
```

**Request Body:**
```json
{
  "query": "What is the main topic of the document?",
  "user_id": "optional-user-id"
}
```

**Example with curl:**
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic of the document?",
    "user_id": "user123"
  }'
```

**Response:**
```json
{
  "status": "success",
  "message": "Query processed successfully",
  "context": "Retrieved relevant text chunks from the vector database..."
}
```

## Project Structure

```
rag/
├── api/                      # API endpoints
│   ├── __init__.py
│   ├── documents.py         # Document upload endpoint
│   ├── health.py            # Health check endpoint
│   └── query.py             # Query processing endpoint
├── core/                     # Core business logic
│   ├── __init__.py
│   ├── text_extractor.py    # PDF text extraction
│   ├── convert_to_chunks.py # Text chunking
│   ├── generate_embedding_on_chunks.py  # Embedding generation
│   ├── store_vector_embeddings.py       # Vector storage
│   ├── retrieve_vector_embeddings.py    # Vector retrieval
│   └── execute_query.py     # Query execution pipeline
├── vector_db/               # Vector database integration
│   ├── __init__.py
│   └── qdrant.py           # Qdrant client and operations
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── logger.py           # Logging configuration
│   └── api_response_format.py  # API response formatting
├── tests/                   # Test suite
│   ├── test_api/           # API endpoint tests
│   └── test_core/          # Core logic tests
├── memory-bank/            # Project documentation
├── main.py                 # Application entry point
├── pyproject.toml          # Project configuration
├── .env.example            # Environment variables template
└── README.md              # This file
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=. --cov-report=html

# Run specific test file
uv run pytest tests/test_api/test_health_endpoint.py
```

### Code Organization

- **Modular Design**: Each component has a single responsibility
- **Type Hints**: Comprehensive type annotations throughout
- **Error Handling**: Robust error handling with detailed logging
- **Async Support**: FastAPI's async capabilities for better performance

### Logging

The system uses structured logging with configurable levels:

```python
# In your code
from utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Processing document")
logger.error("Error occurred", exc_info=True)
```

Configure logging via environment variables:
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR, CRITICAL
- `LOG_FILE`: Optional file path for log output
- `USE_JSON_LOGGING`: Enable JSON-formatted logs for production

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | FastAPI | High-performance async API framework |
| **Server** | Uvicorn | ASGI server for FastAPI |
| **Vector Database** | Qdrant | Efficient vector similarity search |
| **Embeddings** | Sentence Transformers | High-quality text embeddings |
| **PDF Processing** | PyMuPDF | Text extraction from PDFs |
| **Testing** | pytest | Unit and integration testing |
| **Package Manager** | uv | Fast Python package management |
| **Error Tracking** | Sentry | Production error monitoring (optional) |

## Configuration

### Embedding Model

The system uses `sentence-transformers` for generating embeddings. The default model can be configured in `core/generate_embedding_on_chunks.py`.

### Chunking Strategy

Text chunking parameters can be adjusted in `core/convert_to_chunks.py` to optimize for your use case:
- Chunk size
- Overlap between chunks
- Splitting strategy

### Vector Search

Qdrant search parameters can be tuned in `core/retrieve_vector_embeddings.py`:
- Number of results to retrieve
- Similarity threshold
- Search filters

## Performance Considerations

- **Embedding Generation**: Computationally expensive; consider batch processing
- **Vector Search**: Sub-second response times with proper Qdrant configuration
- **Memory Usage**: Scales with document collection size
- **Concurrency**: FastAPI handles multiple concurrent requests efficiently

## Security Notes

For production deployment:

1. **Add Authentication**: Implement API key or OAuth authentication
2. **Input Validation**: Validate file uploads and query inputs
3. **Rate Limiting**: Implement rate limiting to prevent abuse
4. **HTTPS**: Use HTTPS in production
5. **Environment Variables**: Never commit `.env` file to version control

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Vector search powered by [Qdrant](https://qdrant.tech/)
- Embeddings from [Sentence Transformers](https://www.sbert.net/)

## Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This is a foundational RAG system. For production use, consider adding:
- Authentication and authorization
- Caching layer for frequently accessed documents
- Async document processing with task queues
- Monitoring and observability tools
- Horizontal scaling capabilities
