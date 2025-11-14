"""Text processing and cleaning utilities."""

import re                               # Regular expressions for text processing
import hashlib                          # For hashing text for deduplication
import unicodedata                      # For unicode normalization
from typing import List, Dict, Any, Set # Type hints
import logging                          # Logging
from pathlib import Path                # File path handling

logger = logging.getLogger(__name__)


class TextCleaner:
    """Clean and normalize text."""
    
    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text: unicode, whitespace, etc."""
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Remove control characters except newlines and tabs
        text = ''.join(
            char for char in text 
            if unicodedata.category(char)[0] != 'C' or char in '\n\t'
        )
        
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove zero-width characters
        text = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff]', '', text)
        
        # Trim whitespace from lines
        text = '\n'.join(line.strip() for line in text.split('\n'))
        
        # Final trim
        return text.strip()


class Deduplicator:
    """Handle text deduplication."""
    
    def __init__(self):
        self.seen_hashes: Set[str] = set()
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate."""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        if text_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(text_hash)
        return False
    
    def reset(self):
        """Clear seen hashes."""
        self.seen_hashes.clear()


class LanguageDetector:
    """Detect text language."""
    
    @staticmethod
    def detect(text: str, sample_size: int = 1000) -> str:
        """Detect language of text."""
        # Use first N characters for speed
        sample = text[:sample_size]
        
        try:
            from langdetect import detect, LangDetectException
            try:
                return detect(sample)
            except LangDetectException:
                return 'unknown'
        except ImportError:
            logger.warning("langdetect not installed, using simple detection")
            return LanguageDetector._simple_detect(sample)
    
    @staticmethod
    def _simple_detect(text: str) -> str:
        """Simple English detection fallback."""
        english_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 
            'and', 'or', 'but', 'in', 'of', 'to', 'for'
        }
        
        words = set(text.lower().split()[:100])
        
        if len(words & english_words) > 3:
            return 'en'
        
        return 'unknown'


class TextChunker:
    """Chunk text into token-based segments."""
    
    def __init__(self, min_size: int = 250, max_size: int = 400):
        self.min_size = min_size
        self.max_size = max_size
    
    def tokenize(self, text: str) -> List[str]:
        """Simple word-based tokenization."""
        return re.findall(r'\b\w+\b|[^\w\s]', text)
    
    def chunk(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """Chunk text into segments with metadata."""
        tokens = self.tokenize(text)
        chunks = []
        i = 0
        chunk_id = 0
        
        while i < len(tokens):
            # Extract chunk
            chunk_tokens = tokens[i:i + self.max_size]
            
            # Try to break at sentence boundary
            if len(chunk_tokens) >= self.min_size and i + self.max_size < len(tokens):
                search_start = int(len(chunk_tokens) * 0.8)
                for j in range(len(chunk_tokens) - 1, search_start, -1):
                    if chunk_tokens[j] in {'.', '!', '?', ';'}:
                        chunk_tokens = chunk_tokens[:j + 1]
                        break
            
            # Skip very short chunks
            if len(chunk_tokens) < 10:
                i += len(chunk_tokens)
                continue
            
            chunk_text = ' '.join(chunk_tokens)
            
            chunks.append({
                'id': f"{Path(source_file).stem}_chunk_{chunk_id}",
                'text': chunk_text,
                'token_count': len(chunk_tokens),
                'source_file': source_file,
                'chunk_index': chunk_id
            })
            
            chunk_id += 1
            i += len(chunk_tokens)
        
        return chunks
