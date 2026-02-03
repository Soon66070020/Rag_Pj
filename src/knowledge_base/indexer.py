"""Document indexing pipeline for Weaviate vector database.

This module provides the main indexing pipeline that orchestrates:
1. Document loading (PDF, text files)
2. Semantic chunking with Thai support
3. Metadata extraction
4. Embedding generation (BGE-M3)
5. Indexing to Weaviate

The pipeline handles batching, progress tracking, and error recovery.
"""

import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict
import time

from src.core.types import Document
from src.core.exceptions import IndexingError, DocumentLoadError, ChunkingError, EmbeddingError
from src.knowledge_base.loader import load_document, LoadedPage
from src.knowledge_base.chunker import SemanticChunker, TextChunk
from src.knowledge_base.metadata_extractor import MetadataExtractor
from src.knowledge_base.embedding_generator import BGEEmbeddingGenerator
from src.database.manager import WeaviateManager
from config.settings import Settings


logger = logging.getLogger(__name__)


class DocumentIndexer:
    """Main document indexing pipeline.

    Orchestrates the complete pipeline from raw files to indexed
    vectors in Weaviate:

    Raw Files → Load → Chunk → Extract Metadata → Generate Embeddings → Index to Weaviate

    Attributes:
        settings: Configuration settings.
        chunker: Semantic chunker for Thai text.
        metadata_extractor: Metadata extraction engine.
        embedding_generator: BGE-M3 embedding generator.
        db_manager: Weaviate database manager.
        batch_size: Batch size for indexing.

    Example:
        >>> indexer = DocumentIndexer(settings)
        >>> indexer.index_file(Path("medication_guide.pdf"))
        >>> # Documents are now searchable in Weaviate
    """

    def __init__(
        self,
        settings: Settings,
        db_manager: Optional[WeaviateManager] = None,
        embedding_generator: Optional[BGEEmbeddingGenerator] = None,
        batch_size: int = 100
    ):
        """Initialize document indexer.

        Args:
            settings: Configuration settings.
            db_manager: Weaviate manager. Creates new if None.
            embedding_generator: Embedding generator. Creates new if None.
            batch_size: Batch size for indexing operations.
        """
        self.settings = settings
        self.batch_size = batch_size

        # Initialize chunker from config
        chunking_config = settings.model_config.get('chunking', {})
        self.chunker = SemanticChunker(
            chunk_size=chunking_config.get('chunk_size', 512),
            chunk_overlap=chunking_config.get('chunk_overlap', 50),
            min_chunk_size=chunking_config.get('min_chunk_size', 100),
            max_chunk_size=chunking_config.get('max_chunk_size', 800)
        )

        # Initialize metadata extractor
        self.metadata_extractor = MetadataExtractor(
            auto_infer_category=True,
            extract_keywords=True
        )

        # Initialize embedding generator
        if embedding_generator is None:
            embedding_config = settings.model_config.get('embedding', {})
            self.embedding_generator = BGEEmbeddingGenerator(
                model_name=embedding_config.get('model_name', 'BAAI/bge-m3'),
                device=embedding_config.get('device', 'cuda'),
                batch_size=embedding_config.get('batch_size', 32),
                use_fp16=embedding_config.get('use_fp16', True),
                max_length=embedding_config.get('max_length', 512)
            )
        else:
            self.embedding_generator = embedding_generator

        # Initialize database manager
        if db_manager is None:
            from src.database.connector import get_connector
            connector = get_connector(settings)
            from src.database.manager import get_manager
            self.db_manager = get_manager(connector, settings)
        else:
            self.db_manager = db_manager

        logger.info(
            f"Initialized DocumentIndexer with batch_size={batch_size}, "
            f"collection={self.db_manager.collection_name}"
        )

    def index_file(
        self,
        file_path: Path,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Index a single document file.

        Complete pipeline: Load → Chunk → Extract Metadata → Embed → Index

        Args:
            file_path: Path to document file (PDF, TXT, MD).
            show_progress: Whether to log progress messages.

        Returns:
            Dictionary with indexing statistics:
            {
                'file': str,
                'pages': int,
                'chunks': int,
                'indexed': int,
                'failed': int,
                'time_seconds': float
            }

        Raises:
            IndexingError: If indexing fails.

        Example:
            >>> stats = indexer.index_file(Path("guide.pdf"))
            >>> print(f"Indexed {stats['indexed']} chunks")
        """
        if not file_path.exists():
            raise IndexingError(f"File not found: {file_path}")

        logger.info(f"Starting indexing pipeline for: {file_path}")
        start_time = time.time()

        try:
            # Step 1: Load document
            if show_progress:
                logger.info(f"[1/5] Loading document: {file_path.name}")

            pages = load_document(file_path)

            if not pages:
                raise IndexingError(f"No content extracted from {file_path}")

            logger.info(f"Loaded {len(pages)} pages from {file_path.name}")

            # Step 2: Chunk document
            if show_progress:
                logger.info(f"[2/5] Chunking document into semantic segments")

            chunks = self._chunk_pages(pages, file_path)

            if not chunks:
                raise IndexingError(f"No chunks created from {file_path}")

            logger.info(f"Created {len(chunks)} chunks")

            # Step 3: Extract metadata
            if show_progress:
                logger.info(f"[3/5] Extracting metadata and categories")

            documents = self._extract_metadata(chunks, file_path)
            logger.info(f"Extracted metadata for {len(documents)} chunks")

            # Step 4: Generate embeddings
            if show_progress:
                logger.info(f"[4/5] Generating BGE-M3 embeddings")

            documents = self._generate_embeddings(documents)
            logger.info(f"Generated embeddings for {len(documents)} documents")

            # Step 5: Index to Weaviate
            if show_progress:
                logger.info(f"[5/5] Indexing to Weaviate")

            indexed_count = self._index_to_weaviate(documents)

            elapsed = time.time() - start_time

            stats = {
                'file': str(file_path),
                'pages': len(pages),
                'chunks': len(chunks),
                'indexed': indexed_count,
                'failed': len(documents) - indexed_count,
                'time_seconds': round(elapsed, 2)
            }

            logger.info(
                f"Successfully indexed {file_path.name}: "
                f"{indexed_count} chunks in {elapsed:.2f}s"
            )

            return stats

        except (DocumentLoadError, ChunkingError, EmbeddingError) as e:
            error_msg = f"Failed to index {file_path}: {e}"
            logger.error(error_msg)
            raise IndexingError(error_msg) from e

    def index_directory(
        self,
        directory: Path,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True
    ) -> Dict[str, Any]:
        """Index all documents in a directory.

        Args:
            directory: Directory containing documents.
            file_extensions: List of extensions to process (e.g., ['.pdf', '.txt']).
                If None, processes all supported formats.
            recursive: Whether to search subdirectories.

        Returns:
            Dictionary with overall statistics:
            {
                'total_files': int,
                'successful': int,
                'failed': int,
                'total_chunks': int,
                'time_seconds': float,
                'file_stats': List[Dict]
            }

        Example:
            >>> stats = indexer.index_directory(Path("data/raw"))
            >>> print(f"Indexed {stats['successful']}/{stats['total_files']} files")
        """
        if not directory.exists() or not directory.is_dir():
            raise IndexingError(f"Directory not found: {directory}")

        if file_extensions is None:
            file_extensions = ['.pdf', '.txt', '.md', '.markdown']

        logger.info(f"Indexing directory: {directory}")
        logger.info(f"Extensions: {file_extensions}, Recursive: {recursive}")

        start_time = time.time()

        # Find all files
        files = []
        if recursive:
            for ext in file_extensions:
                files.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in file_extensions:
                files.extend(directory.glob(f"*{ext}"))

        if not files:
            logger.warning(f"No files found in {directory}")
            return {
                'total_files': 0,
                'successful': 0,
                'failed': 0,
                'total_chunks': 0,
                'time_seconds': 0.0,
                'file_stats': []
            }

        logger.info(f"Found {len(files)} files to index")

        # Index each file
        successful = 0
        failed = 0
        total_chunks = 0
        file_stats = []

        for i, file_path in enumerate(files, start=1):
            logger.info(f"Processing file {i}/{len(files)}: {file_path.name}")

            try:
                stats = self.index_file(file_path, show_progress=False)
                successful += 1
                total_chunks += stats['indexed']
                file_stats.append(stats)

            except Exception as e:
                logger.error(f"Failed to index {file_path.name}: {e}")
                failed += 1
                file_stats.append({
                    'file': str(file_path),
                    'error': str(e),
                    'indexed': 0
                })

        elapsed = time.time() - start_time

        overall_stats = {
            'total_files': len(files),
            'successful': successful,
            'failed': failed,
            'total_chunks': total_chunks,
            'time_seconds': round(elapsed, 2),
            'file_stats': file_stats
        }

        logger.info(
            f"Directory indexing complete: {successful}/{len(files)} files, "
            f"{total_chunks} chunks in {elapsed:.2f}s"
        )

        return overall_stats

    def _chunk_pages(
        self,
        pages: List[LoadedPage],
        source_file: Path
    ) -> List[TextChunk]:
        """Chunk pages into semantic segments.

        Args:
            pages: List of loaded pages.
            source_file: Source file path.

        Returns:
            List of text chunks.
        """
        # Convert LoadedPage to dict format for chunker
        page_dicts = [
            {
                'text': page.text,
                'page_number': page.page_number,
                'source_file': str(source_file)
            }
            for page in pages
        ]

        chunks = self.chunker.chunk_pages(page_dicts)
        return chunks

    def _extract_metadata(
        self,
        chunks: List[TextChunk],
        source_file: Path
    ) -> List[Document]:
        """Extract metadata and create Document objects.

        Args:
            chunks: List of text chunks.
            source_file: Source file path.

        Returns:
            List of Document objects with metadata.
        """
        documents = []

        for chunk in chunks:
            # Extract metadata
            metadata = self.metadata_extractor.extract(
                text=chunk.text,
                source_file=str(source_file),
                page_number=chunk.metadata.get('page_number', 1),
                chunk_id=chunk.chunk_id,
                additional_metadata=chunk.metadata
            )

            # Create Document object
            doc = Document(
                id=str(uuid.uuid4()),
                content=chunk.text,
                category=metadata.get('category', 'Post-op Care'),
                source_file=source_file.name,
                page_number=metadata.get('page_number', 1),
                metadata=metadata
            )

            documents.append(doc)

        return documents

    def _generate_embeddings(self, documents: List[Document]) -> List[Document]:
        """Generate embeddings for documents.

        Args:
            documents: List of Document objects.

        Returns:
            Same list with embeddings populated.
        """
        # Extract texts for batch embedding
        texts = [doc.content for doc in documents]

        # Generate embeddings in batches
        embeddings = self.embedding_generator.encode_batch(
            texts,
            batch_size=self.batch_size
        )

        # Populate embeddings in documents
        for doc, embedding in zip(documents, embeddings):
            doc.dense_vector = embedding.get('dense', [])
            doc.sparse_vector = embedding.get('sparse', {})

            # Convert numpy array to list if needed
            if hasattr(doc.dense_vector, 'tolist'):
                doc.dense_vector = doc.dense_vector.tolist()

        return documents

    def _index_to_weaviate(self, documents: List[Document]) -> int:
        """Index documents to Weaviate in batches.

        Args:
            documents: List of Document objects with embeddings.

        Returns:
            Number of successfully indexed documents.
        """
        collection = self.db_manager.get_collection()

        indexed_count = 0
        failed_count = 0

        # Process in batches
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]

            try:
                # Prepare batch data
                with collection.batch.dynamic() as batch_context:
                    for doc in batch:
                        # Prepare properties (metadata + content)
                        properties = {
                            'content': doc.content,
                            'category': doc.category,
                            'source_file': doc.source_file,
                            'page_number': doc.page_number
                        }

                        # Add to batch with vector
                        batch_context.add_object(
                            properties=properties,
                            uuid=doc.id,
                            vector=doc.dense_vector
                        )

                indexed_count += len(batch)

                logger.debug(
                    f"Indexed batch {i // self.batch_size + 1}: "
                    f"{len(batch)} documents"
                )

            except Exception as e:
                logger.error(f"Failed to index batch: {e}")
                failed_count += len(batch)

        logger.info(
            f"Indexing complete: {indexed_count} successful, {failed_count} failed"
        )

        return indexed_count


def index_file(
    file_path: Path,
    settings: Optional[Settings] = None
) -> Dict[str, Any]:
    """Convenience function to index a single file.

    Args:
        file_path: Path to document file.
        settings: Configuration settings. Uses defaults if None.

    Returns:
        Indexing statistics dictionary.

    Example:
        >>> stats = index_file(Path("medication_guide.pdf"))
        >>> print(f"Indexed {stats['indexed']} chunks")
    """
    if settings is None:
        from config.settings import get_settings
        settings = get_settings()

    indexer = DocumentIndexer(settings)
    return indexer.index_file(file_path)


def index_directory(
    directory: Path,
    settings: Optional[Settings] = None,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function to index a directory.

    Args:
        directory: Directory containing documents.
        settings: Configuration settings. Uses defaults if None.
        **kwargs: Additional arguments for index_directory.

    Returns:
        Overall indexing statistics.

    Example:
        >>> stats = index_directory(Path("data/raw"))
        >>> print(f"Indexed {stats['total_chunks']} chunks from "
        ...       f"{stats['successful']} files")
    """
    if settings is None:
        from config.settings import get_settings
        settings = get_settings()

    indexer = DocumentIndexer(settings)
    return indexer.index_directory(directory, **kwargs)
