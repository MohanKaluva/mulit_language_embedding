"""Main pipeline orchestrator."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from config import Config
from loaders import FileLoader
from processor import TextCleaner, Deduplicator, LanguageDetector, TextChunker
from embedder import TextEmbedder  # NEW
from indexer import FAISSIndexer  # NEW
from utils import ensure_directories, find_files, save_jsonl, save_json

logger = logging.getLogger(__name__)


class TextProcessingPipeline:
    """Complete text processing pipeline with embeddings and FAISS indexing."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Initialize existing components
        self.loader = FileLoader()
        self.cleaner = TextCleaner()
        self.deduplicator = Deduplicator()
        self.language_detector = LanguageDetector()
        self.chunker = TextChunker(
            self.config.CHUNK_SIZE_MIN,
            self.config.CHUNK_SIZE_MAX
        )
        
        # NEW: Initialize embedder
        self.embedder = TextEmbedder(
            model_name=self.config.EMBEDDING_MODEL,
            batch_size=self.config.EMBEDDING_BATCH_SIZE,
            use_gpu=self.config.USE_GPU
        )
        
        # NEW: Initialize FAISS indexer
        self.indexer = FAISSIndexer(
            embedding_dim=self.embedder.embedding_dim
        )
        
        # Ensure directories exist
        ensure_directories(
            self.config.RAW_DATA_DIR,
            self.config.PROCESSED_DATA_DIR,
            self.config.INDEX_DIR  # NEW
        )
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'duplicate_files': 0,
            'total_chunks': 0,
            'total_tokens': 0,
            'total_embeddings': 0  # NEW
        }
        
        self.processed_outputs = []
        self.all_chunks = []  # NEW: Store all chunks for indexing
    
    def process_file(self, file_path: Path) -> Optional[List[Dict[str, Any]]]:
        """Process a single file through the pipeline."""
        try:
            logger.info(f"Processing: {file_path.name}")
            
            # 1. Load
            raw_text = self.loader.load(file_path)
            
            if not raw_text or len(raw_text.strip()) < self.config.MIN_TEXT_LENGTH:
                logger.warning(f"Text too short: {file_path.name}")
                return None
            
            # 2. Clean
            normalized_text = self.cleaner.normalize(raw_text)
            
            # 3. Deduplicate
            if self.deduplicator.is_duplicate(normalized_text):
                logger.warning(f"Duplicate detected: {file_path.name}")
                self.stats['duplicate_files'] += 1
                return None
            
            # 4. Detect language
            language = self.language_detector.detect(
                normalized_text,
                self.config.LANGUAGE_DETECTION_SAMPLE_SIZE
            )
            
            # 5. Chunk
            relative_path = str(file_path.relative_to(self.config.RAW_DATA_DIR))
            chunks = self.chunker.chunk(normalized_text, relative_path)
            
            # Add metadata to chunks
            for chunk in chunks:
                chunk['language'] = language
                chunk['processed_at'] = datetime.now().isoformat()
            
            # Update statistics
            self.stats['processed_files'] += 1
            self.stats['total_chunks'] += len(chunks)
            self.stats['total_tokens'] += sum(c['token_count'] for c in chunks)
            
            logger.info(
                f"✓ Processed {file_path.name}: "
                f"{len(chunks)} chunks, language: {language}"
            )
            
            return chunks
            
        except Exception as e:
            logger.error(f"✗ Failed to process {file_path.name}: {str(e)}")
            self.stats['failed_files'] += 1
            return None
    
    def create_metadata(self) -> Dict[str, Any]:
        """Create metadata for the processing run."""
        metadata = {
            'processed_at': datetime.now().isoformat(),
            'statistics': self.stats,
            'config': {
                'chunk_size_min': self.config.CHUNK_SIZE_MIN,
                'chunk_size_max': self.config.CHUNK_SIZE_MAX,
                'embedding_model': self.config.EMBEDDING_MODEL,  # NEW
                'embedding_dim': self.embedder.embedding_dim,  # NEW
            },
            'output_files': self.processed_outputs,
            'index_info': self.indexer.get_stats(),  # NEW
            'languages': {}
        }
        
        # Count languages
        for output in self.processed_outputs:
            lang = output.get('language', 'unknown')
            if lang:
                metadata['languages'][lang] = metadata['languages'].get(lang, 0) + output['chunk_count']
        
        return metadata
    
    def run(self):
        """Run the complete pipeline with embedding and indexing."""
        logger.info("=" * 60)
        logger.info("Starting Text Processing Pipeline with Embeddings")
        logger.info("=" * 60)
        
        # Find all files
        files = find_files(
            self.config.RAW_DATA_DIR,
            self.config.SUPPORTED_EXTENSIONS
        )
        
        self.stats['total_files'] = len(files)
        
        if not files:
            logger.warning(f"No files found in {self.config.RAW_DATA_DIR}")
            return
        
        logger.info(f"Found {len(files)} file(s) to process")
        
        # Process each file
        for file_path in files:
            chunks = self.process_file(file_path)
            if chunks:
                # Save chunks to JSONL
                output_name = f"{file_path.stem}_clean.jsonl"
                output_path = self.config.PROCESSED_DATA_DIR / output_name
                save_jsonl(chunks, output_path)
                
                # Track output
                self.processed_outputs.append({
                    'input_file': file_path.name,
                    'output_file': output_name,
                    'chunk_count': len(chunks),
                    'token_count': sum(c['token_count'] for c in chunks),
                    'language': chunks[0].get('language', 'unknown') if chunks else 'unknown'
                })
                
                # NEW: Collect all chunks for embedding
                self.all_chunks.extend(chunks)
        
        if not self.all_chunks:
            logger.warning("No chunks generated")
            return
        
        # NEW: Generate embeddings for all chunks
        logger.info("\n" + "=" * 60)
        logger.info("Generating Embeddings")
        logger.info("=" * 60)
        embeddings = self.embedder.embed_chunks(self.all_chunks)
        self.stats['total_embeddings'] = len(embeddings)
        
        # NEW: Add embeddings to FAISS index
        logger.info("\n" + "=" * 60)
        logger.info("Building FAISS Index")
        logger.info("=" * 60)
        self.indexer.add_embeddings(embeddings, self.all_chunks)
        
        # NEW: Save FAISS index
        index_path = self.config.INDEX_DIR / self.config.FAISS_INDEX_FILE
        metadata_path = self.config.INDEX_DIR / self.config.FAISS_METADATA_FILE
        self.indexer.save(index_path, metadata_path)
        
        # Save pipeline metadata
        metadata = self.create_metadata()
        metadata_path = self.config.PROCESSED_DATA_DIR / self.config.METADATA_FILE
        save_json(metadata, metadata_path)
        
        return self.stats