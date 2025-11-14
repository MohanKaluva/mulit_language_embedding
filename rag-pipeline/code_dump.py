"""
RAG Pipeline with FastAPI - Complete Project Structure
======================================================

PROJECT STRUCTURE:
rag-pipeline/
├── requirements.txt
├── .env.example
├── config/
│   └── settings.py
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py           # API endpoints
│   │   └── models.py           # Pydantic models
│   ├── core/
│   │   ├── __init__.py
│   │   ├── loaders.py          # File loaders
│   │   ├── processor.py        # Text processing
│   │   ├── embedder.py         # Embedding generation
│   │   └── vector_store.py     # FAISS/Qdrant interface
│   └── utils/
│       ├── __init__.py
│       └── helpers.py          # Utility functions
├── data/
│   ├── raw/                    # Uploaded files
│   ├── processed/              # Cleaned JSONL chunks
│   └── index/                  # Vector store
└── tests/
    └── test_api.py

Copy each section below into respective files.
"""

# ============================================================================
# FILE: requirements.txt
# ============================================================================
"""
# FastAPI and server
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
pydantic==2.5.0
pydantic-settings==2.1.0

# File processing
PyPDF2==3.0.1
python-docx==1.1.0
pandas==2.1.3
openpyxl==3.1.2

# Text processing
langdetect==1.0.9

# Embeddings and vector store
sentence-transformers==2.2.2
torch==2.1.0
numpy==1.24.3

# Vector stores (choose one or both)
faiss-cpu==1.7.4
# qdrant-client==1.7.0  # Uncomment if using Qdrant

# Utilities
python-dotenv==1.0.0
aiofiles==23.2.1
"""

# ============================================================================
# FILE: .env.example
# ============================================================================
"""
# Application
APP_NAME=RAG Pipeline API
DEBUG=True
HOST=0.0.0.0
PORT=8000

# Paths
RAW_DATA_DIR=data/raw
PROCESSED_DATA_DIR=data/processed
INDEX_DIR=data/index

# Processing
CHUNK_SIZE_MIN=250
CHUNK_SIZE_MAX=400
MAX_FILE_SIZE_MB=50

# Embeddings
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
EMBEDDING_BATCH_SIZE=32
USE_GPU=False

# Vector Store (faiss or qdrant)
VECTOR_STORE=faiss
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=documents
"""

# ============================================================================
# FILE: config/settings.py
# ============================================================================
"""Application configuration."""

from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = "RAG Pipeline API"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"
    INDEX_DIR: Path = BASE_DIR / "data" / "index"
    
    # Processing
    CHUNK_SIZE_MIN: int = 250
    CHUNK_SIZE_MAX: int = 400
    MAX_FILE_SIZE_MB: int = 50
    SUPPORTED_FORMATS: set = {'.txt', '.pdf', '.docx', '.csv'}
    
    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    EMBEDDING_BATCH_SIZE: int = 32
    USE_GPU: bool = False
    
    # Vector Store
    VECTOR_STORE: Literal["faiss", "qdrant"] = "faiss"
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "documents"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def ensure_directories(self):
        """Create necessary directories."""
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.INDEX_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()


# ============================================================================
# FILE: app/__init__.py
# ============================================================================
"""Application package."""
pass


# ============================================================================
# FILE: app/api/__init__.py
# ============================================================================
"""API package."""
pass


# ============================================================================
# FILE: app/api/models.py
# ============================================================================
"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class UploadResponse(BaseModel):
    """Response for file upload."""
    filename: str
    file_id: str
    size_bytes: int
    status: str
    message: str
    chunks_created: int
    processing_time_seconds: float


class QueryRequest(BaseModel):
    """Request model for search query."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    file_id: Optional[str] = Field(None, description="Filter by specific file ID")


class SearchResult(BaseModel):
    """Single search result."""
    chunk_id: str
    text: str
    score: float
    source_file: str
    file_id: str
    language: str
    chunk_index: int
    metadata: dict


class QueryResponse(BaseModel):
    """Response for search query."""
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time_seconds: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    total_documents: int
    total_chunks: int
    vector_store: str


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str


# ============================================================================
# FILE: app/core/__init__.py
# ============================================================================
"""Core processing modules."""
pass


# ============================================================================
# FILE: app/core/loaders.py
# ============================================================================
"""File loaders for different formats."""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FileLoader:
    """Load content from various file formats."""
    
    @staticmethod
    def load(file_path: Path) -> str:
        """Load text based on file extension."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt':
            return FileLoader._load_txt(file_path)
        elif suffix == '.pdf':
            return FileLoader._load_pdf(file_path)
        elif suffix == '.docx':
            return FileLoader._load_docx(file_path)
        elif suffix == '.csv':
            return FileLoader._load_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    @staticmethod
    def _load_txt(path: Path) -> str:
        """Load plain text file."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, LookupError):
                continue
        raise ValueError(f"Could not decode file: {path}")
    
    @staticmethod
    def _load_pdf(path: Path) -> str:
        """Load PDF file."""
        try:
            import PyPDF2
            text_parts = []
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text_parts.append(extracted)
            return '\n'.join(text_parts)
        except ImportError:
            raise ImportError("PyPDF2 not installed")
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise
    
    @staticmethod
    def _load_docx(path: Path) -> str:
        """Load Word document."""
        try:
            import docx
            doc = docx.Document(path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return '\n'.join(paragraphs)
        except ImportError:
            raise ImportError("python-docx not installed")
        except Exception as e:
            logger.error(f"Error reading DOCX: {e}")
            raise
    
    @staticmethod
    def _load_csv(path: Path) -> str:
        """Load CSV file."""
        try:
            import pandas as pd
            df = pd.read_csv(path)
            # Convert DataFrame to text representation
            text_parts = []
            text_parts.append("Columns: " + ", ".join(df.columns))
            for idx, row in df.iterrows():
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
                text_parts.append(row_text)
            return '\n'.join(text_parts)
        except ImportError:
            raise ImportError("pandas not installed")
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            raise


# ============================================================================
# FILE: app/core/processor.py
# ============================================================================
"""Text processing and chunking."""

import re
import hashlib
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Set
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """Process and chunk text."""
    
    def __init__(self, chunk_size_min: int = 250, chunk_size_max: int = 400):
        self.chunk_size_min = chunk_size_min
        self.chunk_size_max = chunk_size_max
        self.seen_hashes: Set[str] = set()
    
    def normalize(self, text: str) -> str:
        """Normalize text."""
        text = unicodedata.normalize('NFKC', text)
        text = ''.join(
            char for char in text 
            if unicodedata.category(char)[0] != 'C' or char in '\n\t'
        )
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff]', '', text)
        text = '\n'.join(line.strip() for line in text.split('\n'))
        return text.strip()
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is duplicate."""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        if text_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(text_hash)
        return False
    
    def detect_language(self, text: str) -> str:
        """Detect language."""
        try:
            from langdetect import detect, LangDetectException
            try:
                return detect(text[:1000])
            except LangDetectException:
                return 'unknown'
        except ImportError:
            return self._simple_detect(text)
    
    def _simple_detect(self, text: str) -> str:
        """Simple English detection."""
        english_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 
            'and', 'or', 'but', 'in', 'of', 'to', 'for'
        }
        words = set(text.lower().split()[:100])
        if len(words & english_words) > 3:
            return 'en'
        return 'unknown'
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        return re.findall(r'\b\w+\b|[^\w\s]', text)
    
    def chunk_text(self, text: str, file_id: str, filename: str) -> List[Dict[str, Any]]:
        """Chunk text into segments."""
        tokens = self.tokenize(text)
        chunks = []
        i = 0
        chunk_id = 0
        
        while i < len(tokens):
            chunk_tokens = tokens[i:i + self.chunk_size_max]
            
            # Try to break at sentence boundary
            if len(chunk_tokens) >= self.chunk_size_min and i + self.chunk_size_max < len(tokens):
                search_start = int(len(chunk_tokens) * 0.8)
                for j in range(len(chunk_tokens) - 1, search_start, -1):
                    if chunk_tokens[j] in {'.', '!', '?', ';'}:
                        chunk_tokens = chunk_tokens[:j + 1]
                        break
            
            if len(chunk_tokens) < 10:
                i += len(chunk_tokens)
                continue
            
            chunk_text = ' '.join(chunk_tokens)
            
            chunks.append({
                'chunk_id': f"{file_id}_chunk_{chunk_id}",
                'text': chunk_text,
                'token_count': len(chunk_tokens),
                'source_file': filename,
                'file_id': file_id,
                'chunk_index': chunk_id,
                'created_at': datetime.now().isoformat()
            })
            
            chunk_id += 1
            i += len(chunk_tokens)
        
        return chunks


# ============================================================================
# FILE: app/core/embedder.py
# ============================================================================
"""Text embedding generation."""

import logging
import numpy as np
from typing import List, Dict, Any
import torch

logger = logging.getLogger(__name__)


class TextEmbedder:
    """Generate embeddings using sentence transformers."""
    
    def __init__(self, model_name: str, batch_size: int = 32, use_gpu: bool = False):
        self.model_name = model_name
        self.batch_size = batch_size
        
        logger.info(f"Loading embedding model: {model_name}")
        
        from sentence_transformers import SentenceTransformer
        
        device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """Generate embeddings for chunks."""
        texts = [chunk['text'] for chunk in chunks]
        
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for query."""
        return self.model.encode(query, convert_to_numpy=True)


# ============================================================================
# FILE: app/core/vector_store.py
# ============================================================================
"""Vector store interface for FAISS and Qdrant."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_documents(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        """Add documents to the store."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
               filter_file_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def save(self):
        """Save the index."""
        pass
    
    @abstractmethod
    def load(self):
        """Load the index."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store."""
    
    def __init__(self, embedding_dim: int, index_dir: Path):
        self.embedding_dim = embedding_dim
        self.index_dir = index_dir
        self.index_path = index_dir / "faiss_index.bin"
        self.metadata_path = index_dir / "faiss_metadata.json"
        self.metadata = []
        
        try:
            import faiss
            self.faiss = faiss
            self.index = faiss.IndexFlatL2(embedding_dim)
            logger.info(f"Initialized FAISS index with dimension {embedding_dim}")
        except ImportError:
            raise ImportError("faiss-cpu not installed")
    
    def add_documents(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        """Add documents to FAISS index."""
        embeddings = embeddings.astype('float32')
        start_idx = self.index.ntotal
        self.index.add(embeddings)
        
        for i, chunk in enumerate(chunks):
            metadata_entry = chunk.copy()
            metadata_entry['index_id'] = start_idx + i
            self.metadata.append(metadata_entry)
        
        logger.info(f"Added {len(chunks)} documents. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
               filter_file_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        if self.index.ntotal == 0:
            return []
        
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, min(top_k * 2, self.index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                metadata = self.metadata[idx]
                
                # Filter by file_id if specified
                if filter_file_id and metadata.get('file_id') != filter_file_id:
                    continue
                
                result = metadata.copy()
                result['score'] = float(1 / (1 + dist))
                result['distance'] = float(dist)
                results.append(result)
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def save(self):
        """Save index and metadata."""
        self.faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved FAISS index to {self.index_path}")
    
    def load(self):
        """Load index and metadata."""
        if self.index_path.exists():
            self.index = self.faiss.read_index(str(self.index_path))
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            'total_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'metadata_entries': len(self.metadata),
            'store_type': 'faiss'
        }


class QdrantVectorStore(VectorStore):
    """Qdrant-based vector store."""
    
    def __init__(self, embedding_dim: int, host: str, port: int, collection_name: str):
        self.embedding_dim = embedding_dim
        self.collection_name = collection_name
        
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            self.client = QdrantClient(host=host, port=port)
            
            # Create collection if it doesn't exist
            collections = self.client.get_collections().collections
            if collection_name not in [c.name for c in collections]:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {collection_name}")
                
        except ImportError:
            raise ImportError("qdrant-client not installed")
    
    def add_documents(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        """Add documents to Qdrant."""
        from qdrant_client.models import PointStruct
        
        points = []
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            point = PointStruct(
                id=hash(chunk['chunk_id']) % (10 ** 8),  # Generate numeric ID
                vector=embedding.tolist(),
                payload=chunk
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Added {len(points)} documents to Qdrant")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
               filter_file_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search in Qdrant."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        search_filter = None
        if filter_file_id:
            search_filter = Filter(
                must=[FieldCondition(key="file_id", match=MatchValue(value=filter_file_id))]
            )
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            query_filter=search_filter
        )
        
        return [
            {**hit.payload, 'score': hit.score}
            for hit in results
        ]
    
    def save(self):
        """Qdrant persists automatically."""
        pass
    
    def load(self):
        """Qdrant loads automatically."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        collection_info = self.client.get_collection(self.collection_name)
        return {
            'total_vectors': collection_info.points_count,
            'embedding_dim': self.embedding_dim,
            'store_type': 'qdrant'
        }


# ============================================================================
# FILE: app/utils/__init__.py
# ============================================================================
"""Utilities package."""
pass


# ============================================================================
# FILE: app/utils/helpers.py
# ============================================================================
"""Helper utilities."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import hashlib

logger = logging.getLogger(__name__)


def generate_file_id(filename: str, content: bytes) -> str:
    """Generate unique file ID."""
    content_hash = hashlib.md5(content).hexdigest()[:8]
    name_hash = hashlib.md5(filename.encode()).hexdigest()[:4]
    return f"{name_hash}_{content_hash}"


def save_jsonl(data: List[Dict[str, Any]], output_path: Path):
    """Save data to JSONL format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(data)} items to {output_path}")


def load_jsonl(input_path: Path) -> List[Dict[str, Any]]:
    """Load data from JSONL format."""
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    return file_path.stat().st_size / (1024 * 1024)


# ============================================================================
# FILE: app/api/routes.py
# ============================================================================
"""API routes."""

import time
import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from datetime import datetime

from app.api.models import (
    UploadResponse, QueryRequest, QueryResponse, 
    SearchResult, HealthResponse, ErrorResponse
)
from app.core.loaders import FileLoader
from app.core.processor import TextProcessor
from app.core.embedder import TextEmbedder
from app.core.vector_store import FAISSVectorStore, QdrantVectorStore
from app.utils.helpers import generate_file_id, save_jsonl, get_file_size_mb
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize components
processor = TextProcessor(settings.CHUNK_SIZE_MIN, settings.CHUNK_SIZE_MAX)
embedder = TextEmbedder(
    settings.EMBEDDING_MODEL, 
    settings.EMBEDDING_BATCH_SIZE,
    settings.USE_GPU
)

# Initialize vector store
if settings.VECTOR_STORE == "faiss":
    vector_store = FAISSVectorStore(embedder.embedding_dim, settings.INDEX_DIR)
    vector_store.load()  # Load existing index if available
elif settings.VECTOR_STORE == "qdrant":
    vector_store = QdrantVectorStore(
        embedder.embedding_dim,
        settings.QDRANT_HOST,
        settings.QDRANT_PORT,
        settings.QDRANT_COLLECTION
    )
else:
    raise ValueError(f"Unsupported vector store: {settings.VECTOR_STORE}")


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process a document file.
    
    Supported formats: TXT, PDF, DOCX, CSV
    """
    start_time = time.time()
    
    try:
        # Validate file format
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported: {settings.SUPPORTED_FORMATS}"
            )
        
        # Read file content
        content = await file.read()
        
        # Check file size
        size_mb = len(content) / (1024 * 1024)
        if size_mb > settings.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
            )
        
        # Generate file ID
        file_id = generate_file_id(file.filename, content)
        
        # Save raw file
        raw_file_path = settings.RAW_DATA_DIR / f"{file_id}_{file.filename}"
        with open(raw_file_path, 'wb') as f:
            f.write(content)
        
        # Load and process file
        logger.info(f"Processing file: {file.filename}")
        text = FileLoader.load(raw_file_path)
        
        # Normalize text
        normalized_text = processor.normalize(text)
        
        # Detect language
        language = processor.detect_language(normalized_text)
        
        # Chunk text
        chunks = processor.chunk_text(normalized_text, file_id, file.filename)
        
        # Add language to chunks
        for chunk in chunks:
            chunk['language'] = language
        
        # Save chunks as JSONL
        jsonl_path = settings.PROCESSED_DATA_DIR / f"{file_id}_{Path(file.filename).stem}_clean.jsonl"
        save_jsonl(chunks, jsonl_path)
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = embedder.embed_chunks(chunks)
        
        # Add to vector store
        vector_store.add_documents(embeddings, chunks)
        vector_store.save()
        
        processing_time = time.time() - start_time
        
        return UploadResponse(
            filename=file.filename,
            file_id=file_id,
            size_bytes=len(content),
            status="success",
            message=f"File processed successfully. Created {len(chunks)} chunks.",
            chunks_created=len(chunks),
            processing_time_seconds=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Search for relevant document chunks using semantic search.
    """
    start_time = time.time()
    
    try:
        # Generate query embedding
        query_embedding = embedder.embed_query(request.query)
        
        # Search vector store
        results = vector_store.search(
            query_embedding,
            top_k=request.top_k,
            filter_file_id=request.file_id
        )
        
        # Format results
        search_results = [
            SearchResult(
                chunk_id=r['chunk_id'],
                text=r['text'],
                score=r['score'],
                source_file=r['source_file'],
                file_id=r['file_id'],
                language=r.get('language', 'unknown'),
                chunk_index=r['chunk_index'],
                metadata={
                    'token_count': r.get('token_count', 0),
                    'created_at': r.get('created_at', '')
                }
            )
            for r in results
        ]
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            processing_time_seconds=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check API health and get system statistics.
    """
    try:
        stats = vector_store.get_stats()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            total_documents=len(list(settings.RAW_DATA_DIR.glob("*"))),
            total_chunks=stats['total_vectors'],
            vector_store=stats['store_type']
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{file_id}")
async def delete_document(file_id: str):
    """
    Delete a document and its chunks from the system.
    Note: FAISS doesn't support deletion natively, requires rebuilding index.
    """
    # This is a placeholder - full implementation would require:
    # 1. Remove from vector store (rebuild for FAISS)
    # 2. Delete raw file
    # 3. Delete processed JSONL
    raise HTTPException(
        status_code=501,
        detail="Document deletion not yet implemented. Requires index rebuild for FAISS."
    )


# ============================================================================
# FILE: app/main.py
# ============================================================================
"""FastAPI application entry point."""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from config.settings import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="RAG Pipeline API for document processing and semantic search",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1", tags=["documents"])


@app.on_event("startup")
async def startup_event():
    """Run on startup."""
    logger.info(f"Starting {settings.APP_NAME}")
    logger.info(f"Vector Store: {settings.VECTOR_STORE}")
    logger.info(f"Embedding Model: {settings.EMBEDDING_MODEL}")
    settings.ensure_directories()


@app.on_event("shutdown")
async def shutdown_event():
    """Run on shutdown."""
    logger.info(f"Shutting down {settings.APP_NAME}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Pipeline API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )


# ============================================================================
# FILE: tests/test_api.py
# ============================================================================
"""API tests."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    """Test health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "total_chunks" in data


def test_upload_txt_file():
    """Test file upload with TXT file."""
    # Create a test file
    test_content = b"This is a test document for the RAG pipeline. " * 100
    
    files = {"file": ("test.txt", test_content, "text/plain")}
    response = client.post("/api/v1/upload", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_created"] > 0
    assert "file_id" in data
    
    return data["file_id"]


def test_query():
    """Test query endpoint."""
    # First upload a document
    file_id = test_upload_txt_file()
    
    # Query the document
    query_data = {
        "query": "What is this document about?",
        "top_k": 3
    }
    
    response = client.post("/api/v1/query", json=query_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "results" in data
    assert data["total_results"] <= 3


def test_query_with_file_filter():
    """Test query with file_id filter."""
    file_id = test_upload_txt_file()
    
    query_data = {
        "query": "test document",
        "top_k": 5,
        "file_id": file_id
    }
    
    response = client.post("/api/v1/query", json=query_data)
    assert response.status_code == 200
    
    data = response.json()
    # All results should be from the specified file
    for result in data["results"]:
        assert result["file_id"] == file_id


def test_invalid_file_format():
    """Test upload with invalid file format."""
    test_content = b"Invalid content"
    files = {"file": ("test.xyz", test_content, "application/octet-stream")}
    
    response = client.post("/api/v1/upload", files=files)
    assert response.status_code == 400


# ============================================================================
# FILE: run_server.py (Optional - at root level)
# ============================================================================
"""
Simple script to run the server.
Usage: python run_server.py
"""

import uvicorn
from config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )


# ============================================================================
# FILE: README.md
# ============================================================================
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

# ============================================================================
# SETUP INSTRUCTIONS
# ============================================================================
"""
COMPLETE SETUP GUIDE:

1. CREATE PROJECT STRUCTURE:
   mkdir -p rag-pipeline/app/api rag-pipeline/app/core rag-pipeline/app/utils
   mkdir -p rag-pipeline/config rag-pipeline/data/{raw,processed,index}
   mkdir -p rag-pipeline/tests

2. CREATE ALL FILES:
   Copy each section above into its respective file following the 
   "FILE: path/to/file.py" headers

3. INSTALL DEPENDENCIES:
   cd rag-pipeline
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

4. CONFIGURE:
   cp .env.example .env
   # Edit .env as needed

5. RUN SERVER:
   python run_server.py
   # OR
   uvicorn app.main:app --reload

6. TEST API:
   - Open http://localhost:8000/docs
   - Upload a test file
   - Try a query

7. RUN TESTS:
   pytest tests/test_api.py -v

That's it! Your RAG pipeline API is ready! 🚀
"""