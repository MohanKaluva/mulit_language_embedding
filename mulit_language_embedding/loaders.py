"""File loaders for different formats."""

from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)    #argument __name__ is a special in-built variable that holds the current module (file) name


class FileLoader:
    """Load text from various file formats."""
    
    @staticmethod
    def load(file_path: Path) -> str:
        """Load text based on file extension."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt':
            return FileLoader._load_txt(file_path)
        elif suffix == '.pdf':
            return FileLoader._load_pdf(file_path)
        elif suffix in ['.doc', '.docx']:
            return FileLoader._load_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    @staticmethod
    def _load_txt(path: Path) -> str:
        """Load plain text file with encoding detection."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, LookupError):
                continue
        
        raise ValueError(f"Could not decode file: {path}")
    
    @staticmethod
    def _load_pdf(path: Path) -> str:
        """Load PDF file."""
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("PyPDF2 not installed. Run: pip install PyPDF2")
        
        text_parts = []
        try:
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text_parts.append(extracted)
        except Exception as e:
            logger.error(f"Error reading PDF {path}: {e}")
            raise
        
        return '\n'.join(text_parts)
    
    @staticmethod
    def _load_docx(path: Path) -> str:
        """Load Word document."""
        try:
            import docx
        except ImportError:
            raise ImportError("python-docx not installed. Run: pip install python-docx")
        
        try:
            doc = docx.Document(path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return '\n'.join(paragraphs)
        except Exception as e:
            logger.error(f"Error reading DOCX {path}: {e}")
            raise
