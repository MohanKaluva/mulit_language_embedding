"""Main entry point for the pipeline."""

import logging
from pathlib import Path

from config import Config
from pipeline import TextProcessingPipeline
from embedder import TextEmbedder  # NEW
from indexer import FAISSIndexer  # NEW
from utils import setup_logging, print_summary, find_files

logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample data for demonstration."""
    sample_dir = Path('data/raw')
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample 1: NLP text
    sample_text_1 = """
    Natural Language Processing (NLP) is a fascinating field of artificial intelligence.
    It focuses on the interaction between computers and human language.
    
    NLP enables computers to understand, interpret, and generate human language in valuable ways.
    This technology powers many applications we use daily, from chatbots to translation services.
    
    Machine learning has revolutionized NLP, allowing systems to learn from vast amounts of text data.
    Modern NLP models can perform complex tasks like sentiment analysis, named entity recognition,
    and text summarization with impressive accuracy.
    
    The future of NLP looks promising with continued advances in deep learning and transformer models.
    These models can process context and semantics in ways that were impossible just a few years ago.
    
    From language translation to content generation, NLP is transforming how we interact with technology.
    The applications are endless and continue to grow as the technology matures.
    """ * 3
    
    # Sample 2: AI text
    sample_text_2 = """
    Artificial Intelligence has transformed the technological landscape in unprecedented ways.
    Machine learning algorithms now power everything from recommendation systems to autonomous vehicles.
    
    Deep learning, a subset of machine learning, uses neural networks with multiple layers to process data.
    These networks can learn hierarchical representations of data, making them incredibly powerful.
    
    Computer vision allows machines to interpret and understand visual information from the world.
    Applications range from facial recognition to medical image analysis and quality control in manufacturing.
    
    The ethical implications of AI are being actively discussed by researchers and policymakers worldwide.
    Questions about bias, privacy, and accountability are central to responsible AI development.
    """ * 3
    
    # Create multiple sample files
    samples = [
        ('sample_nlp_text.txt', sample_text_1),
        ('sample_ai_text.txt', sample_text_2)
    ]
    
    for filename, content in samples:
        sample_file = sample_dir / filename
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Created sample file: {sample_file}")


def demonstrate_search(config: Config):
    """Demonstrate similarity search using the FAISS index."""
    logger.info("\n" + "=" * 60)
    logger.info("Demonstrating Similarity Search")
    logger.info("=" * 60)
    
    try:
        # Load the embedder
        embedder = TextEmbedder(
            model_name=config.EMBEDDING_MODEL,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            use_gpu=config.USE_GPU
        )
        
        # Load the index
        indexer = FAISSIndexer(embedding_dim=embedder.embedding_dim)
        index_path = config.INDEX_DIR / config.FAISS_INDEX_FILE
        metadata_path = config.INDEX_DIR / config.FAISS_METADATA_FILE
        
        if not index_path.exists():
            logger.warning("Index not found. Run the pipeline first.")
            return
        
        indexer.load(index_path, metadata_path)
        
        # Example queries
        queries = [
            "What is Dota 2",
            "who is ICE Frog?",
            "Is Dota 2 free?"
        ]
        
        for query in queries:
            print(f"\nQuery: '{query}'")
            print("-" * 60)
            
            # Generate query embedding
            query_embedding = embedder.embed_query(query)
            
            # Search
            results = indexer.search(query_embedding, k=3)
            
            # Display results
            for i, result in enumerate(results, 1):
                print(f"\nResult {i} (Score: {result['similarity_score']:.4f}):")
                print(f"  Source: {result['source_file']}")
                print(f"  Chunk ID: {result['id']}")
                print(f"  Text preview: {result['text'][:150]}...")
        
    except Exception as e:
        logger.error(f"Search demonstration failed: {e}")


def main():
    """Main function to run the pipeline."""
    # Setup logging
    config = Config()
    setup_logging(config.LOG_LEVEL, config.LOG_FORMAT)
    
    # Check if data/raw has any files
    existing_files = find_files(config.RAW_DATA_DIR, config.SUPPORTED_EXTENSIONS)
    
    if not existing_files:
        # Only create sample data if no files exist
        logger.info("No files found in data/raw. Creating sample data...")
        create_sample_data()
    else:
        logger.info(f"Found {len(existing_files)} existing file(s) in data/raw")
    
    # Run pipeline
    pipeline = TextProcessingPipeline(config)
    stats = pipeline.run()
    
    # Print summary
    if stats:
        print_summary(stats, pipeline.processed_outputs)
        
        # NEW: Show index statistics
        print("\n" + "=" * 60)
        print("FAISS Index Information:")
        print("=" * 60)
        index_stats = pipeline.indexer.get_stats()
        print(f"Total vectors indexed:  {index_stats['total_vectors']}")
        print(f"Embedding dimension:    {index_stats['embedding_dim']}")
        print(f"Metadata entries:       {index_stats['metadata_entries']}")
        print(f"Index location:         {config.INDEX_DIR / config.FAISS_INDEX_FILE}")
        print("=" * 60)
        
        # Show sample output from first file
        if pipeline.processed_outputs:
            first_output = pipeline.processed_outputs[0]
            output_file = config.PROCESSED_DATA_DIR / first_output['output_file']
            
            if output_file.exists():
                import json
                print("\nSample output (first chunk from first file):")
                print("-" * 60)
                with open(output_file, 'r', encoding='utf-8') as f:
                    first_chunk = json.loads(f.readline())
                    print(json.dumps(first_chunk, indent=2, ensure_ascii=False))
                print("-" * 60)
        
        # NEW: Demonstrate search functionality
        demonstrate_search(config)


if __name__ == "__main__":
    main()