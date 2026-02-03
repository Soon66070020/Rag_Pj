"""Knowledge base management module.

This module provides functionality for:
- Loading documents (PDF, text files)
- Semantic chunking with Thai support
- Metadata extraction
- Embedding generation (BGE-M3)
- Indexing to Weaviate

Main Classes:
    - DocumentLoader: Load documents from files
    - SemanticChunker: Chunk Thai text semantically
    - MetadataExtractor: Extract document metadata
    - BGEEmbeddingGenerator: Generate embeddings
    - DocumentIndexer: Complete indexing pipeline

Convenience Functions:
    - load_document: Load a single document
    - chunk_document: Chunk a document
    - extract_metadata: Extract metadata
    - encode_documents: Generate embeddings
    - index_file: Index a single file
    - index_directory: Index a directory

Example:
    >>> from src.knowledge_base import index_file
    >>> stats = index_file(Path("medication_guide.pdf"))
    >>> print(f"Indexed {stats['indexed']} chunks")
"""

from src.knowledge_base.loader import (
    DocumentLoader,
    PDFLoader,
    TextLoader,
    LoadedPage,
    get_loader,
    load_document
)

from src.knowledge_base.chunker import (
    SemanticChunker,
    TextChunk,
    chunk_document
)

from src.knowledge_base.metadata_extractor import (
    MetadataExtractor,
    extract_metadata,
    batch_extract_metadata
)

from src.knowledge_base.embedding_generator import (
    BGEEmbeddingGenerator,
    create_embedding_generator,
    encode_documents
)

from src.knowledge_base.indexer import (
    DocumentIndexer,
    index_file,
    index_directory
)


__all__ = [
    # Loaders
    'DocumentLoader',
    'PDFLoader',
    'TextLoader',
    'LoadedPage',
    'get_loader',
    'load_document',

    # Chunking
    'SemanticChunker',
    'TextChunk',
    'chunk_document',

    # Metadata
    'MetadataExtractor',
    'extract_metadata',
    'batch_extract_metadata',

    # Embeddings
    'BGEEmbeddingGenerator',
    'create_embedding_generator',
    'encode_documents',

    # Indexing
    'DocumentIndexer',
    'index_file',
    'index_directory',
]
