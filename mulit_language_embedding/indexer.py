"""FAISS indexing for efficient similarity search."""

import logging
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndexer:
    """FAISS-based vector index for similarity search."""
    
    def __init__(self, embedding_dim: int):
        """
        Initialize FAISS index.
        
        Args:
            embedding_dim: Dimension of embedding vectors
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata = []
        
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError(
                "faiss not installed. "
                "Run: pip install faiss-cpu (or faiss-gpu for GPU support)"
            )
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize a new FAISS index."""
        # Using IndexFlatL2 for exact search (good for small to medium datasets)
        # For larger datasets, consider IndexIVFFlat or IndexHNSWFlat
        self.index = self.faiss.IndexFlatL2(self.embedding_dim)
        logger.info(f"Initialized FAISS index with dimension {self.embedding_dim}")
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        """
        Add embeddings to the index.
        
        Args:
            embeddings: numpy array of embeddings (n x embedding_dim)
            chunks: List of chunk metadata dictionaries
        """
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} "
                f"doesn't match index dimension {self.embedding_dim}"
            )
        
        # Ensure embeddings are float32 (FAISS requirement)
        embeddings = embeddings.astype('float32')
        
        # Add to index
        start_idx = self.index.ntotal
        self.index.add(embeddings)
        
        # Store metadata with index references
        for i, chunk in enumerate(chunks):
            metadata_entry = chunk.copy()
            metadata_entry['index_id'] = start_idx + i
            self.metadata.append(metadata_entry)
        
        logger.info(f"✓ Added {len(chunks)} embeddings to index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector (embedding_dim,)
            k: Number of results to return
            
        Returns:
            List of dictionaries with chunk metadata and similarity scores
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Ensure query is 2D array and float32
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:  # Valid index
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(1 / (1 + dist))  # Convert distance to similarity
                result['distance'] = float(dist)
                results.append(result)
        
        return results
    
    def save(self, index_path: Path, metadata_path: Path):
        """Save index and metadata to disk."""
        # Save FAISS index
        self.faiss.write_index(self.index, str(index_path))
        logger.info(f"✓ Saved FAISS index to {index_path}")
        
        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved metadata to {metadata_path}")
    
    def load(self, index_path: Path, metadata_path: Path):
        """Load index and metadata from disk."""
        # Load FAISS index
        self.index = self.faiss.read_index(str(index_path))
        logger.info(f"✓ Loaded FAISS index from {index_path}")
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        logger.info(f"✓ Loaded {len(self.metadata)} metadata entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'embedding_dim': self.embedding_dim,
            'metadata_entries': len(self.metadata)
        }
