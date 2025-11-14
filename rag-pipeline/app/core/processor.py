"""Text processing and chunking."""

import re
import hashlib
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Set
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """Process and chunk text."""
    
    def __init__(self, chunk_size_min: int = 250, chunk_size_max: int = 400):
        self.chunk_size_min = chunk_size_min
        self.chunk_size_max = chunk_size_max
        self.seen_hashes: Set[str] = set()
    
    def normalize(self, text: str) -> str:
        """Normalize text."""
        text = unicodedata.normalize('NFKC', text)
        text = ''.join(
            char for char in text 
            if unicodedata.category(char)[0] != 'C' or char in '\n\t'
        )
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff]', '', text)
        text = '\n'.join(line.strip() for line in text.split('\n'))
        return text.strip()
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is duplicate."""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        if text_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(text_hash)
        return False
    
    def detect_language(self, text: str) -> str:
        """Detect language."""
        try:
            from langdetect import detect, LangDetectException
            try:
                return detect(text[:1000])
            except LangDetectException:
                return 'unknown'
        except ImportError:
            return self._simple_detect(text)
    
    def _simple_detect(self, text: str) -> str:
        """Simple English detection."""
        english_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 
            'and', 'or', 'but', 'in', 'of', 'to', 'for'
        }
        words = set(text.lower().split()[:100])
        if len(words & english_words) > 3:
            return 'en'
        return 'unknown'
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        return re.findall(r'\b\w+\b|[^\w\s]', text)
    
    def chunk_text(self, text: str, file_id: str, filename: str) -> List[Dict[str, Any]]:
        """Chunk text into segments."""
        tokens = self.tokenize(text)
        chunks = []
        i = 0
        chunk_id = 0
        
        while i < len(tokens):
            chunk_tokens = tokens[i:i + self.chunk_size_max]
            
            # Try to break at sentence boundary
            if len(chunk_tokens) >= self.chunk_size_min and i + self.chunk_size_max < len(tokens):
                search_start = int(len(chunk_tokens) * 0.8)
                for j in range(len(chunk_tokens) - 1, search_start, -1):
                    if chunk_tokens[j] in {'.', '!', '?', ';'}:
                        chunk_tokens = chunk_tokens[:j + 1]
                        break
            
            if len(chunk_tokens) < 10:
                i += len(chunk_tokens)
                continue
            
            chunk_text = ' '.join(chunk_tokens)
            
            chunks.append({
                'chunk_id': f"{file_id}_chunk_{chunk_id}",
                'text': chunk_text,
                'token_count': len(chunk_tokens),
                'source_file': filename,
                'file_id': file_id,
                'chunk_index': chunk_id,
                'created_at': datetime.now().isoformat()
            })
            
            chunk_id += 1
            i += len(chunk_tokens)
        
        return chunks