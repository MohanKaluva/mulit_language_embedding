"""Helper utilities."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import hashlib

logger = logging.getLogger(__name__)


def generate_file_id(filename: str, content: bytes) -> str:
    """Generate unique file ID."""
    content_hash = hashlib.md5(content).hexdigest()[:8]
    name_hash = hashlib.md5(filename.encode()).hexdigest()[:4]
    return f"{name_hash}_{content_hash}"


def save_jsonl(data: List[Dict[str, Any]], output_path: Path):
    """Save data to JSONL format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(data)} items to {output_path}")


def load_jsonl(input_path: Path) -> List[Dict[str, Any]]:
    """Load data from JSONL format."""
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    return file_path.stat().st_size / (1024 * 1024)