"""Database setup script for initializing Weaviate schema.

This script:
1. Connects to Weaviate database
2. Creates the MedicalGuideline collection if it doesn't exist
3. Displays collection statistics

Usage:
    python scripts/setup_database.py
    python scripts/setup_database.py --reset  # Delete and recreate collection
    python scripts/setup_database.py --check  # Just check status
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from src.database.connector import get_connector
from src.database.manager import get_manager
from src.core.exceptions import DatabaseError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_database_status(manager):
    """Check and display database status.

    Args:
        manager: WeaviateManager instance.
    """
    logger.info("Checking database status...")

    # Check connection
    if not manager.connector.is_connected():
        logger.error("❌ Not connected to Weaviate")
        return False

    logger.info("✅ Connected to Weaviate")

    # Get server version
    try:
        version = manager.connector.get_server_version()
        logger.info(f"Weaviate version: {version}")
    except Exception as e:
        logger.warning(f"Could not retrieve version: {e}")

    # List all collections
    collections = manager.list_all_collections()
    logger.info(f"Total collections: {len(collections)}")

    if collections:
        logger.info("Collections:")
        for coll_name in collections:
            logger.info(f"  - {coll_name}")

    # Check if our collection exists
    collection_name = manager.collection_name
    if manager.collection_exists(collection_name):
        logger.info(f"✅ Collection '{collection_name}' exists")

        # Get statistics
        try:
            stats = manager.get_collection_stats(collection_name)
            logger.info(f"  Objects count: {stats['object_count']}")
        except Exception as e:
            logger.warning(f"Could not get stats: {e}")

    else:
        logger.warning(f"⚠️  Collection '{collection_name}' does not exist")

    return True


def create_schema(manager, force=False):
    """Create database schema.

    Args:
        manager: WeaviateManager instance.
        force: If True, recreate even if exists.
    """
    collection_name = manager.collection_name
    logger.info(f"Creating collection '{collection_name}'...")

    try:
        if force:
            logger.warning("Force mode: will delete existing collection")
            if manager.collection_exists(collection_name):
                logger.warning(f"Deleting existing collection '{collection_name}'")
                manager.delete_schema(collection_name, confirm=True)

        # Create schema
        created = manager.create_schema(skip_if_exists=not force)

        if created:
            logger.info(f"✅ Successfully created collection '{collection_name}'")

            # Display schema info
            try:
                stats = manager.get_collection_stats(collection_name)
                logger.info(f"Collection ready with {stats['object_count']} objects")
            except Exception as e:
                logger.warning(f"Could not get initial stats: {e}")

        else:
            logger.info(f"Collection '{collection_name}' already exists (skipped)")

        return True

    except DatabaseError as e:
        logger.error(f"❌ Failed to create schema: {e}")
        return False


def reset_collection(manager):
    """Reset collection by deleting and recreating.

    Args:
        manager: WeaviateManager instance.
    """
    collection_name = manager.collection_name
    logger.warning(f"⚠️  RESETTING collection '{collection_name}'")
    logger.warning("This will DELETE ALL DATA in the collection!")

    try:
        # Confirm with user
        response = input("Are you sure? Type 'yes' to confirm: ")
        if response.lower() != 'yes':
            logger.info("Reset cancelled")
            return False

        # Reset collection
        manager.reset_collection(collection_name, confirm=True)
        logger.info(f"✅ Successfully reset collection '{collection_name}'")

        return True

    except DatabaseError as e:
        logger.error(f"❌ Failed to reset collection: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup Weaviate database schema for RAG system"
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Delete and recreate the collection (WARNING: deletes all data)'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Only check database status, do not create schema'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recreate schema even if it exists'
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("RAG System - Database Setup")
    logger.info("=" * 70)

    try:
        # Load settings
        logger.info("Loading configuration...")
        settings = get_settings()
        logger.info(f"Weaviate URL: {settings.weaviate_url}")

        # Connect to database
        logger.info("Connecting to Weaviate...")
        connector = get_connector(settings)

        if not connector.is_connected():
            logger.error("❌ Failed to connect to Weaviate")
            logger.error("Make sure Weaviate is running and accessible")
            sys.exit(1)

        logger.info("✅ Connected successfully")

        # Create manager
        manager = get_manager(connector, settings)
        logger.info(f"Target collection: {manager.collection_name}")

        # Handle different modes
        if args.reset:
            # Reset mode
            success = reset_collection(manager)
            if not success:
                sys.exit(1)

        elif args.check:
            # Check only mode
            check_database_status(manager)

        else:
            # Normal mode: create schema
            success = create_schema(manager, force=args.force)
            if not success:
                sys.exit(1)

            # Show final status
            logger.info("")
            check_database_status(manager)

        logger.info("=" * 70)
        logger.info("✅ Database setup completed successfully")
        logger.info("=" * 70)

    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
