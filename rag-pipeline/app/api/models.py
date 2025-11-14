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