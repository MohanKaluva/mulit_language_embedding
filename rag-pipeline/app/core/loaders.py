"""File loaders for different formats."""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FileLoader:
    """Load content from various file formats."""
    
    @staticmethod
    def load(file_path: Path) -> str:
        """Load text based on file extension."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt':
            return FileLoader._load_txt(file_path)
        elif suffix == '.pdf':
            return FileLoader._load_pdf(file_path)
        elif suffix == '.docx':
            return FileLoader._load_docx(file_path)
        elif suffix == '.csv':
            return FileLoader._load_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    @staticmethod
    def _load_txt(path: Path) -> str:
        """Load plain text file."""
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
            text_parts = []
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text_parts.append(extracted)
            return '\n'.join(text_parts)
        except ImportError:
            raise ImportError("PyPDF2 not installed")
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise
    
    @staticmethod
    def _load_docx(path: Path) -> str:
        """Load Word document."""
        try:
            import docx
            doc = docx.Document(path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return '\n'.join(paragraphs)
        except ImportError:
            raise ImportError("python-docx not installed")
        except Exception as e:
            logger.error(f"Error reading DOCX: {e}")
            raise
    
    @staticmethod
    def _load_csv(path: Path) -> str:
        """Load CSV file with robust error handling."""
        try:
            import pandas as pd
            
            # Try multiple strategies to read the CSV
            strategies = [
                # Strategy 1: Standard read
                {'error_bad_lines': False, 'warn_bad_lines': True},
                # Strategy 2: Skip bad lines (pandas 1.3+)
                {'on_bad_lines': 'skip'},
                # Strategy 3: No error checking, variable columns
                {'on_bad_lines': 'skip', 'engine': 'python'},
                # Strategy 4: Read as text with flexible delimiter
                {'sep': None, 'engine': 'python', 'on_bad_lines': 'skip'},
            ]
            
            df = None
            for i, strategy in enumerate(strategies):
                try:
                    df = pd.read_csv(path, **strategy)
                    logger.info(f"CSV loaded with strategy {i+1}")
                    break
                except Exception as e:
                    logger.debug(f"Strategy {i+1} failed: {e}")
                    continue
            
            # If all strategies fail, try reading as plain text
            if df is None or df.empty:
                logger.warning("All pandas strategies failed, reading as plain text")
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            # Convert DataFrame to text representation
            text_parts = []
            
            # Add column headers
            text_parts.append("Columns: " + ", ".join(str(col) for col in df.columns))
            text_parts.append("-" * 80)
            
            # Add rows
            for idx, row in df.iterrows():
                # Handle NaN values
                row_items = []
                for col, val in row.items():
                    if pd.notna(val):
                        row_items.append(f"{col}: {val}")
                
                if row_items:  # Only add non-empty rows
                    row_text = " | ".join(row_items)
                    text_parts.append(row_text)
            
            # Add summary
            text_parts.append("-" * 80)
            text_parts.append(f"Total rows: {len(df)}, Total columns: {len(df.columns)}")
            
            result = '\n'.join(text_parts)
            
            if not result.strip():
                raise ValueError("CSV file resulted in empty content")
            
            return result
            
        except ImportError:
            raise ImportError("pandas not installed")
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            # Last resort: read as plain text
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if content.strip():
                        logger.info("CSV read as plain text fallback")
                        return content
            except Exception as text_error:
                logger.error(f"Plain text fallback also failed: {text_error}")
            raise