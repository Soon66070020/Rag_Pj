"""Document ingestion script for the RAG system.

This script provides a CLI to ingest documents into Weaviate:
1. Loads documents (PDF, text files)
2. Chunks them semantically with Thai support
3. Generates BGE-M3 embeddings
4. Indexes to Weaviate

Usage:
    # Ingest a single file
    python scripts/ingest_documents.py --file data/raw/medication_guide.pdf

    # Ingest a directory
    python scripts/ingest_documents.py --directory data/raw

    # Ingest with custom settings
    python scripts/ingest_documents.py --directory data/raw --batch-size 50

    # Check status only (don't ingest)
    python scripts/ingest_documents.py --status
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from src.database.connector import get_connector
from src.database.manager import get_manager
from src.knowledge_base.indexer import DocumentIndexer
from src.core.exceptions import IndexingError, DatabaseError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_database_status(settings):
    """Check database connection and status.

    Args:
        settings: Configuration settings.

    Returns:
        True if database is ready, False otherwise.
    """
    logger.info("Checking database status...")

    try:
        # Connect to database
        connector = get_connector(settings)

        if not connector.is_connected():
            logger.error("❌ Not connected to Weaviate")
            return False

        logger.info("✅ Connected to Weaviate")

        # Get server version
        try:
            version = connector.get_server_version()
            logger.info(f"Weaviate version: {version}")
        except Exception as e:
            logger.warning(f"Could not retrieve version: {e}")

        # Check collection
        manager = get_manager(connector, settings)
        collection_name = manager.collection_name

        if manager.collection_exists(collection_name):
            logger.info(f"✅ Collection '{collection_name}' exists")

            # Get statistics
            try:
                stats = manager.get_collection_stats(collection_name)
                logger.info(f"  Current object count: {stats['object_count']}")
            except Exception as e:
                logger.warning(f"Could not get stats: {e}")

        else:
            logger.warning(f"⚠️  Collection '{collection_name}' does not exist")
            logger.info("Run 'python scripts/setup_database.py' to create it")
            return False

        return True

    except DatabaseError as e:
        logger.error(f"❌ Database error: {e}")
        return False


def ingest_file(file_path: Path, settings, batch_size: int = 100):
    """Ingest a single document file.

    Args:
        file_path: Path to document file.
        settings: Configuration settings.
        batch_size: Batch size for indexing.

    Returns:
        True if successful, False otherwise.
    """
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False

    logger.info("=" * 70)
    logger.info(f"Ingesting file: {file_path}")
    logger.info("=" * 70)

    try:
        # Create indexer
        indexer = DocumentIndexer(settings, batch_size=batch_size)

        # Index file
        stats = indexer.index_file(file_path, show_progress=True)

        # Display results
        logger.info("")
        logger.info("=" * 70)
        logger.info("Ingestion Results:")
        logger.info(f"  File: {stats['file']}")
        logger.info(f"  Pages: {stats['pages']}")
        logger.info(f"  Chunks created: {stats['chunks']}")
        logger.info(f"  Chunks indexed: {stats['indexed']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info(f"  Time: {stats['time_seconds']}s")
        logger.info("=" * 70)

        if stats['indexed'] > 0:
            logger.info(f"✅ Successfully ingested {file_path.name}")
            return True
        else:
            logger.error(f"❌ Failed to ingest {file_path.name}")
            return False

    except IndexingError as e:
        logger.error(f"❌ Ingestion failed: {e}")
        return False


def ingest_directory(
    directory: Path,
    settings,
    batch_size: int = 100,
    extensions: Optional[str] = None,
    recursive: bool = True
):
    """Ingest all documents in a directory.

    Args:
        directory: Directory containing documents.
        settings: Configuration settings.
        batch_size: Batch size for indexing.
        extensions: Comma-separated list of extensions (e.g., ".pdf,.txt").
        recursive: Whether to search subdirectories.

    Returns:
        True if at least one file was successfully ingested.
    """
    if not directory.exists() or not directory.is_dir():
        logger.error(f"Directory not found: {directory}")
        return False

    logger.info("=" * 70)
    logger.info(f"Ingesting directory: {directory}")
    logger.info("=" * 70)

    # Parse extensions
    if extensions:
        file_extensions = [ext.strip() for ext in extensions.split(',')]
        logger.info(f"File extensions: {file_extensions}")
    else:
        file_extensions = None
        logger.info("Processing all supported formats")

    logger.info(f"Recursive: {recursive}")
    logger.info("")

    try:
        # Create indexer
        indexer = DocumentIndexer(settings, batch_size=batch_size)

        # Index directory
        stats = indexer.index_directory(
            directory,
            file_extensions=file_extensions,
            recursive=recursive
        )

        # Display results
        logger.info("")
        logger.info("=" * 70)
        logger.info("Ingestion Summary:")
        logger.info(f"  Total files: {stats['total_files']}")
        logger.info(f"  Successful: {stats['successful']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info(f"  Total chunks indexed: {stats['total_chunks']}")
        logger.info(f"  Total time: {stats['time_seconds']}s")
        logger.info("=" * 70)

        # Show per-file stats
        if stats['file_stats']:
            logger.info("")
            logger.info("Per-file statistics:")
            for file_stat in stats['file_stats']:
                if 'error' in file_stat:
                    logger.info(f"  ❌ {Path(file_stat['file']).name}: {file_stat['error']}")
                else:
                    logger.info(
                        f"  ✅ {Path(file_stat['file']).name}: "
                        f"{file_stat['indexed']} chunks in {file_stat['time_seconds']}s"
                    )

        if stats['successful'] > 0:
            logger.info(f"\n✅ Successfully ingested {stats['successful']} files")
            return True
        else:
            logger.error("\n❌ No files were successfully ingested")
            return False

    except IndexingError as e:
        logger.error(f"❌ Directory ingestion failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into Weaviate for RAG system"
    )

    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--file',
        type=Path,
        help='Path to single document file to ingest'
    )
    input_group.add_argument(
        '--directory',
        type=Path,
        help='Path to directory containing documents'
    )
    input_group.add_argument(
        '--status',
        action='store_true',
        help='Check database status only (do not ingest)'
    )

    # Directory options
    parser.add_argument(
        '--extensions',
        type=str,
        help='Comma-separated file extensions (e.g., ".pdf,.txt")'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not search subdirectories'
    )

    # Indexing options
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for indexing (default: 100)'
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("RAG System - Document Ingestion")
    logger.info("=" * 70)

    try:
        # Load settings
        logger.info("Loading configuration...")
        settings = get_settings()
        logger.info(f"Weaviate URL: {settings.weaviate_url}")
        logger.info(f"Collection: {settings.model_config.get('weaviate', {}).get('collection_name', 'MedicalGuideline')}")

        # Check database status
        if not check_database_status(settings):
            logger.error("\n❌ Database is not ready")
            logger.info("Please ensure:")
            logger.info("1. Weaviate is running")
            logger.info("2. Collection is created (run scripts/setup_database.py)")
            sys.exit(1)

        logger.info("")

        # Handle different modes
        if args.status:
            # Status check only
            logger.info("Status check complete")
            sys.exit(0)

        elif args.file:
            # Ingest single file
            success = ingest_file(args.file, settings, args.batch_size)
            sys.exit(0 if success else 1)

        elif args.directory:
            # Ingest directory
            success = ingest_directory(
                args.directory,
                settings,
                args.batch_size,
                args.extensions,
                not args.no_recursive
            )
            sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("\n\nOperation cancelled by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
