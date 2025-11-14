"""Configuration settings for the text processing pipeline."""
from pathlib import Path

class Config:
    """Pipeline configuration."""
    
    # Directory paths
    RAW_DATA_DIR = Path('data/raw')
    PROCESSED_DATA_DIR = Path('data/processed')
    INDEX_DIR = Path('data/index')
    
    # Chunking parameters
    CHUNK_SIZE_MIN = 250
    CHUNK_SIZE_MAX = 400
    
    # File formats
    SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.doc'}
    
    # Processing parameters
    MIN_TEXT_LENGTH = 10  # Minimum text length to process
    LANGUAGE_DETECTION_SAMPLE_SIZE = 1000  # Chars for language detection
    
    # Output files
    METADATA_FILE = 'metadata.json'

    # Embedding configuration
    EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    # Alternative models:
    # 'sentence-transformers/all-MiniLM-L6-v2' (faster, English only)
    # 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' (faster multilingual)
    
    EMBEDDING_BATCH_SIZE = 32
    USE_GPU = False  # Set to True if you have CUDA GPU
    
    # NEW: FAISS configuration
    FAISS_INDEX_FILE = 'faiss_index.bin'
    FAISS_METADATA_FILE = 'faiss_metadata.json'
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

