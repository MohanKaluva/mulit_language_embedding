"""Utility functions."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = 'INFO', log_format: str = None):
    """Setup logging configuration."""
    if log_format is None:
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format
    )


def ensure_directories(*dirs):
    """Ensure directories exist."""
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)


def find_files(directory: Path, extensions: set) -> List[Path]:
    """Find all files with given extensions."""
    files = []
    for ext in extensions:
        files.extend(directory.rglob(f'*{ext}'))
    return sorted(files)


def save_jsonl(data: List[Dict[str, Any]], output_path: Path):
    """Save data to JSONL format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(data)} items to {output_path}")


def save_json(data: Dict[str, Any], output_path: Path):
    """Save data to JSON format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved metadata to {output_path}")


def print_summary(stats: Dict[str, Any], processed_outputs: List[Dict[str, Any]] = None):
    """Print pipeline summary."""
    print("\n" + "=" * 60)
    print("Pipeline Complete - Summary:")
    print("=" * 60)
    print(f"Total files found:     {stats['total_files']}")
    print(f"Successfully processed: {stats['processed_files']}")
    print(f"Duplicates skipped:    {stats['duplicate_files']}")
    print(f"Failed:                {stats['failed_files']}")
    print(f"Total chunks created:  {stats['total_chunks']}")
    print(f"Total tokens:          {stats['total_tokens']}")
    
    if processed_outputs:
        print("\nOutput Files:")
        print("-" * 60)
        for output in processed_outputs:
            if 'input_file' in output:
                print(f"  {output['input_file']} → {output['output_file']}")
                print(f"    Chunks: {output['chunk_count']}, Tokens: {output['token_count']}, Language: {output['language']}")
            else:
                print(f"  Combined → {output['output_file']}")
                print(f"    Chunks: {output['chunk_count']}, Tokens: {output['token_count']}")
    
    print("=" * 60 + "\n")