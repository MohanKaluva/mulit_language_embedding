"""
# RAG Pipeline API

Production-ready FastAPI application for document processing and semantic search.

## Features

✅ Multi-format support (CSV, TXT, DOCX, PDF)
✅ Automatic text cleaning and normalization
✅ Language detection
✅ Smart chunking (250-400 tokens)
✅ Multilingual embeddings
✅ Vector search (FAISS/Qdrant)
✅ RESTful API with FastAPI
✅ Automatic deduplication

## Quick Start

### 1. Installation

```bash
# Clone/create project directory
mkdir rag-pipeline && cd rag-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create `.env` file:
```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Run Server

```bash
# Method 1: Using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Method 2: Using run script
python run_server.py

# Method 3: Using app.main
python -m app.main
```

### 4. Access API

- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/v1/health

## API Endpoints

### Upload Document
```bash
POST /api/v1/upload

curl -X POST "http://localhost:8000/api/v1/upload" \\
  -H "Content-Type: multipart/form-data" \\
  -F "file=@document.pdf"

Response:
{
  "filename": "document.pdf",
  "file_id": "abc123",
  "size_bytes": 102400,
  "status": "success",
  "message": "File processed successfully. Created 15 chunks.",
  "chunks_created": 15,
  "processing_time_seconds": 2.34
}
```

### Query Documents
```bash
POST /api/v1/query

curl -X POST "http://localhost:8000/api/v1/query" \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "What is machine learning?",
    "top_k": 5,
    "file_id": "abc123"
  }'

Response:
{
  "query": "What is machine learning?",
  "results": [
    {
      "chunk_id": "abc123_chunk_0",
      "text": "Machine learning is...",
      "score": 0.92,
      "source_file": "document.pdf",
      "file_id": "abc123",
      "language": "en",
      "chunk_index": 0,
      "metadata": {...}
    }
  ],
  "total_results": 5,
  "processing_time_seconds": 0.15
}
```

### Health Check
```bash
GET /api/v1/health

curl "http://localhost:8000/api/v1/health"

Response:
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "total_documents": 10,
  "total_chunks": 150,
  "vector_store": "faiss"
}
```

## Project Structure

```
rag-pipeline/
├── app/
│   ├── main.py              # FastAPI app
│   ├── api/
│   │   ├── routes.py        # API endpoints
│   │   └── models.py        # Pydantic models
│   ├── core/
│   │   ├── loaders.py       # File loaders
│   │   ├── processor.py     # Text processing
│   │   ├── embedder.py      # Embeddings
│   │   └── vector_store.py  # Vector DB
│   └── utils/
│       └── helpers.py       # Utilities
├── config/
│   └── settings.py          # Configuration
├── data/
│   ├── raw/                 # Uploaded files
│   ├── processed/           # JSONL chunks
│   └── index/               # Vector index
├── tests/
│   └── test_api.py         # API tests
├── requirements.txt
├── .env
└── README.md
```

## Configuration Options

### Vector Store
- **FAISS** (default): Fast, local vector search
- **Qdrant**: Distributed vector database

To use Qdrant:
```env
VECTOR_STORE=qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### Embedding Models
Default: `paraphrase-multilingual-mpnet-base-v2`

Other options:
- `all-MiniLM-L6-v2` (faster, English only)
- `paraphrase-multilingual-MiniLM-L12-v2` (faster multilingual)

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=app
```

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# Upload file
with open("document.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(f"{BASE_URL}/upload", files=files)
    result = response.json()
    file_id = result["file_id"]

# Query
query_data = {
    "query": "What is this about?",
    "top_k": 5,
    "file_id": file_id
}
response = requests.post(f"{BASE_URL}/query", json=query_data)
results = response.json()

for i, result in enumerate(results["results"], 1):
    print(f"{i}. Score: {result['score']:.3f}")
    print(f"   Text: {result['text'][:100]}...")
```

## Deployment

### Docker (Coming Soon)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Tips
- Use `gunicorn` with `uvicorn` workers
- Enable HTTPS
- Add authentication
- Set up monitoring
- Use external vector DB (Qdrant/Pinecone)
- Add rate limiting

## License
MIT
"""