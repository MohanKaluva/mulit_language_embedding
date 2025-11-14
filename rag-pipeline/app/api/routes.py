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

# Initialize components (moved inside route functions to avoid stale references)
def get_processor():
    return TextProcessor(settings.CHUNK_SIZE_MIN, settings.CHUNK_SIZE_MAX)

def get_embedder():
    return TextEmbedder(
        settings.EMBEDDING_MODEL, 
        settings.EMBEDDING_BATCH_SIZE,
        settings.USE_GPU
    )

def get_vector_store(embedder):
    if settings.VECTOR_STORE == "faiss":
        store = FAISSVectorStore(embedder.embedding_dim, settings.INDEX_DIR)
        try:
            store.load()
            logger.info(f"Loaded existing FAISS index with {store.index.ntotal} vectors")
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
        return store
    elif settings.VECTOR_STORE == "qdrant":
        return QdrantVectorStore(
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
        # Get fresh instances
        processor = get_processor()
        embedder = get_embedder()
        vector_store = get_vector_store(embedder)
        
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
        
        if not text or len(text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="File content is too short or empty"
            )
        
        # Normalize text
        normalized_text = processor.normalize(text)
        
        # Detect language
        language = processor.detect_language(normalized_text)
        
        # Chunk text
        chunks = processor.chunk_text(normalized_text, file_id, file.filename)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Could not create any chunks from the file content"
            )
        
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
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        vector_store.add_documents(embeddings, chunks)
        vector_store.save()
        
        # Verify addition
        stats = vector_store.get_stats()
        logger.info(f"Vector store now has {stats['total_vectors']} vectors")
        
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Search for relevant document chunks using semantic search.
    """
    start_time = time.time()
    
    try:
        # Get fresh instances
        embedder = get_embedder()
        vector_store = get_vector_store(embedder)
        
        # Check if vector store has any data
        stats = vector_store.get_stats()
        logger.info(f"Vector store stats: {stats}")
        
        if stats['total_vectors'] == 0:
            logger.warning("Vector store is empty")
            return QueryResponse(
                query=request.query,
                results=[],
                total_results=0,
                processing_time_seconds=round(time.time() - start_time, 2)
            )
        
        # Generate query embedding
        logger.info(f"Processing query: '{request.query}'")
        query_embedding = embedder.embed_query(request.query)
        logger.info(f"Query embedding shape: {query_embedding.shape}")
        
        # Search vector store
        results = vector_store.search(
            query_embedding,
            top_k=request.top_k,
            filter_file_id=request.file_id
        )
        
        logger.info(f"Found {len(results)} results")
        
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
        logger.error(f"Error processing query: {e}", exc_info=True)
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