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
    CHUNK_SIZE_MIN: int = 150
    CHUNK_SIZE_MAX: int = 250
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