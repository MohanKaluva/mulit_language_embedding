"""Vector store interface for FAISS and Qdrant."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import numpy as np
import faiss

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
            self.index = faiss.IndexFlatIP(embedding_dim)
            logger.info(f"Initialized FAISS index with dimension {embedding_dim}")
        except ImportError:
            raise ImportError("faiss-cpu not installed")
    
    def add_documents(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        """Add documents to FAISS index."""
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} "
                f"doesn't match index dimension {self.embedding_dim}"
            )
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings.astype('float32')
        self.faiss.normalize_L2(embeddings)  # Normalize to unit length
        
        # Add to index
        start_idx = self.index.ntotal
        self.index.add(embeddings)
    
        # Store metadata with index references
        for i, chunk in enumerate(chunks):
            metadata_entry = chunk.copy()
            metadata_entry['index_id'] = start_idx + i
            self.metadata.append(metadata_entry)
    
        logger.info(f"Added {len(chunks)} documents. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               filter_file_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        self.faiss.normalize_L2(query_embedding)
        # Search for more results if filtering
        search_k = top_k * 3 if filter_file_id else top_k
        self.similarities, indices = self.index.search(query_embedding, min(search_k, self.index.ntotal))
        logger.info(f"FAISS returned {len(indices[0])} indices")
        logger.info(f"Distances: {self.similarities[0][:5]}")
        logger.info(f"Indices: {indices[0][:5]}")

        results = []
        for similarity, idx in zip(self.similarities[0], indices[0]):
            if idx < 0:
                continue
            
            if idx >= len(self.metadata):
                logger.warning(f"Index {idx} out of bounds (metadata has {len(self.metadata)} entries)")
                continue
        
            metadata = self.metadata[idx]
            
            # Log first result for debugging
            if len(results) == 0:
                logger.info(f"First result - file_id: {metadata.get('file_id')}, filter: {filter_file_id}")
            
            # Filter by file_id if specified
            if filter_file_id and metadata.get('file_id') != filter_file_id:
                logger.debug(f"Filtered out result from file {metadata.get('file_id')}")
                continue
        
            result = metadata.copy()
            result['score'] = float(similarity)  # Cosine similarity (range: -1 to 1, typically 0 to 1 for similar docs)
            result['cosine_similarity'] = float(similarity)
            results.append(result)
            
            if len(results) >= top_k:
                break
    
        logger.info(f"Returning {len(results)} results after filtering")
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