"""Text embedding using multilingual sentence transformers."""

import logging
import numpy as np
from typing import List, Dict, Any
import torch

logger = logging.getLogger(__name__)


class TextEmbedder:
    """Generate embeddings for text using sentence transformers."""
    
    def __init__(self, model_name: str, batch_size: int = 32, use_gpu: bool = False):
        """
        Initialize the embedder.
        
        Args:
            model_name: HuggingFace model name
            batch_size: Batch size for encoding
            use_gpu: Whether to use GPU if available
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        
        logger.info(f"Loading embedding model: {model_name}")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Set device
            if use_gpu and torch.cuda.is_available():
                device = 'cuda'
                logger.info("Using GPU for embeddings")
            else:
                device = 'cpu'
                logger.info("Using CPU for embeddings")
            
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' key
            
        Returns:
            numpy array of embeddings (n_chunks x embedding_dim)
        """
        texts = [chunk['text'] for chunk in chunks]
        
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        
        # Generate embeddings in batches
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info(f"✓ Generated embeddings: {embeddings.shape}")
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True
        )
        
        return embedding
