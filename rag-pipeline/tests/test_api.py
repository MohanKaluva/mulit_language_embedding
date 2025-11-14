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
