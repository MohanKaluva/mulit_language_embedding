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