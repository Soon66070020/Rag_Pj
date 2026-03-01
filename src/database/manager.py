"""Weaviate database schema and data management module.

This module provides high-level operations for managing Weaviate schemas,
collections, and data operations including:
- Schema creation and deletion
- Data ingestion and batch operations
- Collection inspection and statistics
- Data cleanup and maintenance
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.collections.collection import Collection

from config.settings import Settings
from src.database.connector import WeaviateConnector
from src.core.exceptions import DatabaseError


logger = logging.getLogger(__name__)


class WeaviateManager:
    """High-level manager for Weaviate schema and data operations.

    Provides methods for:
    - Creating and deleting collections
    - Loading schemas from JSON configuration
    - Checking collection existence and statistics
    - Batch data operations
    - Data cleanup

    Attributes:
        connector: WeaviateConnector instance.
        settings: Configuration settings.
        collection_name: Default collection name for operations.
    """

    def __init__(
        self,
        connector: WeaviateConnector,
        settings: Settings,
        collection_name: Optional[str] = None
    ):
        """Initialize the Weaviate manager.

        Args:
            connector: WeaviateConnector instance.
            settings: Settings instance.
            collection_name: Default collection name. If None, uses from settings.
        """
        self.connector = connector
        self.settings = settings
        self.collection_name = collection_name or settings.model_config.get(
            "weaviate", {}
        ).get("collection_name", "MedicalGuideline")

    def load_schema_from_config(self) -> Dict[str, Any]:
        """Load Weaviate schema from JSON configuration file.

        Reads the schema definition from config/weaviate_schema.json.

        Returns:
            Schema dictionary.

        Raises:
            DatabaseError: If schema file not found or invalid JSON.

        Example:
            >>> manager = WeaviateManager(connector, settings)
            >>> schema = manager.load_schema_from_config()
            >>> print(schema['class'])
            'MedicalGuideline'
        """
        schema_path = self.settings.config_dir / "weaviate_schema.json"

        if not schema_path.exists():
            raise DatabaseError(f"Schema file not found: {schema_path}")

        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)

            logger.info(f"Loaded schema from {schema_path}")
            return schema

        except json.JSONDecodeError as e:
            raise DatabaseError(f"Invalid JSON in schema file: {e}") from e
        except Exception as e:
            raise DatabaseError(f"Failed to load schema: {e}") from e

    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """Check if a collection exists in Weaviate.

        Args:
            collection_name: Collection name to check. Uses default if None.

        Returns:
            True if collection exists, False otherwise.

        Example:
            >>> if not manager.collection_exists():
            ...     manager.create_schema()
        """
        collection_name = collection_name or self.collection_name

        try:
            client = self.connector.get_client()
            return client.collections.exists(collection_name)

        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False

    def create_schema(self, schema: Optional[Dict[str, Any]] = None, skip_if_exists: bool = True) -> bool:
        """Create Weaviate collection schema.

        Creates a new collection based on the schema definition.
        Supports both v3 and v4 Weaviate API formats.

        Args:
            schema: Schema dictionary. If None, loads from config file.
            skip_if_exists: If True, skip creation if collection already exists.

        Returns:
            True if schema was created, False if skipped.

        Raises:
            DatabaseError: If schema creation fails.

        Example:
            >>> manager.create_schema()
            >>> # Collection created successfully
        """
        # Load schema if not provided
        if schema is None:
            schema = self.load_schema_from_config()

        collection_name = schema.get("class", self.collection_name)

        # Check if collection already exists
        if skip_if_exists and self.collection_exists(collection_name):
            logger.info(f"Collection '{collection_name}' already exists, skipping creation")
            return False

        try:
            client = self.connector.get_client()

            logger.info(f"Creating collection: {collection_name}")

            # Parse properties from schema
            properties = []
            for prop in schema.get("properties", []):
                # Convert dataType list to single DataType enum
                data_type_str = prop["dataType"][0].lower()

                # Map data types
                data_type_map = {
                    "text": DataType.TEXT,
                    "text[]": DataType.TEXT_ARRAY,
                    "int": DataType.INT,
                    "number": DataType.NUMBER,
                    "boolean": DataType.BOOL,
                    "date": DataType.DATE,
                }

                data_type = data_type_map.get(data_type_str, DataType.TEXT)
                is_text_type = data_type == DataType.TEXT

                # Create Property object
                # Note: index_searchable only works with TEXT data types
                property_config = Property(
                    name=prop["name"],
                    data_type=data_type,
                    description=prop.get("description", ""),
                    skip_vectorization=prop.get("skipVectorization", False),
                    index_filterable=prop.get("indexFilterable", True),
                    index_searchable=prop.get("indexSearchable", is_text_type) if is_text_type else False,
                )
                properties.append(property_config)

            # Get vector index config
            vector_config = schema.get("vectorIndexConfig", {})
            distance_metric_str = vector_config.get("distance", "cosine").lower()

            # Map distance metric string to enum
            distance_metric_map = {
                "cosine": VectorDistances.COSINE,
                "dot": VectorDistances.DOT,
                "l2-squared": VectorDistances.L2_SQUARED,
                "l2": VectorDistances.L2_SQUARED,
                "manhattan": VectorDistances.MANHATTAN,
                "hamming": VectorDistances.HAMMING,
            }
            distance_metric = distance_metric_map.get(distance_metric_str, VectorDistances.COSINE)

            # Create collection
            collection = client.collections.create(
                name=collection_name,
                description=schema.get("description", ""),
                properties=properties,
                vectorizer_config=Configure.Vectorizer.none(),  # External vectorization
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=distance_metric
                ),
            )

            logger.info(f"Successfully created collection: {collection_name}")
            return True

        except Exception as e:
            error_msg = f"Failed to create schema for '{collection_name}': {e}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

    def delete_schema(self, collection_name: Optional[str] = None, confirm: bool = False) -> bool:
        """Delete a Weaviate collection and all its data.

        WARNING: This operation is irreversible and will delete all data.

        Args:
            collection_name: Collection name to delete. Uses default if None.
            confirm: Must be True to proceed with deletion (safety check).

        Returns:
            True if collection was deleted, False if not found.

        Raises:
            DatabaseError: If deletion fails.

        Example:
            >>> # Delete with explicit confirmation
            >>> manager.delete_schema(confirm=True)
        """
        if not confirm:
            raise DatabaseError(
                "Must set confirm=True to delete schema. "
                "This operation will delete all data."
            )

        collection_name = collection_name or self.collection_name

        if not self.collection_exists(collection_name):
            logger.warning(f"Collection '{collection_name}' does not exist")
            return False

        try:
            client = self.connector.get_client()
            client.collections.delete(collection_name)

            logger.warning(f"Deleted collection: {collection_name}")
            return True

        except Exception as e:
            error_msg = f"Failed to delete collection '{collection_name}': {e}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

    def get_collection(self, collection_name: Optional[str] = None) -> Collection:
        """Get a Weaviate collection object.

        Args:
            collection_name: Collection name. Uses default if None.

        Returns:
            Collection object for data operations.

        Raises:
            DatabaseError: If collection doesn't exist.

        Example:
            >>> collection = manager.get_collection()
            >>> objects = collection.query.fetch_objects(limit=10)
        """
        collection_name = collection_name or self.collection_name

        if not self.collection_exists(collection_name):
            raise DatabaseError(f"Collection '{collection_name}' does not exist")

        try:
            client = self.connector.get_client()
            return client.collections.get(collection_name)

        except Exception as e:
            raise DatabaseError(f"Failed to get collection: {e}") from e

    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about a collection.

        Args:
            collection_name: Collection name. Uses default if None.

        Returns:
            Dictionary with collection statistics (object count, etc.).

        Raises:
            DatabaseError: If collection doesn't exist or query fails.

        Example:
            >>> stats = manager.get_collection_stats()
            >>> print(f"Total objects: {stats['object_count']}")
        """
        collection_name = collection_name or self.collection_name

        try:
            collection = self.get_collection(collection_name)

            # Get aggregate count
            aggregate_result = collection.aggregate.over_all(total_count=True)
            object_count = aggregate_result.total_count

            stats = {
                "collection_name": collection_name,
                "object_count": object_count,
                "exists": True,
            }

            logger.info(f"Collection '{collection_name}' has {object_count} objects")
            return stats

        except Exception as e:
            error_msg = f"Failed to get collection stats: {e}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

    def delete_all_objects(
        self,
        collection_name: Optional[str] = None,
        confirm: bool = False
    ) -> int:
        """Delete all objects from a collection (keeps schema).

        WARNING: This operation is irreversible.

        Args:
            collection_name: Collection name. Uses default if None.
            confirm: Must be True to proceed (safety check).

        Returns:
            Number of objects deleted.

        Raises:
            DatabaseError: If deletion fails.

        Example:
            >>> count = manager.delete_all_objects(confirm=True)
            >>> print(f"Deleted {count} objects")
        """
        if not confirm:
            raise DatabaseError(
                "Must set confirm=True to delete all objects. "
                "This operation is irreversible."
            )

        collection_name = collection_name or self.collection_name

        try:
            # Get current count
            stats = self.get_collection_stats(collection_name)
            initial_count = stats["object_count"]

            if initial_count == 0:
                logger.info(f"Collection '{collection_name}' is already empty")
                return 0

            # Delete all objects
            collection = self.get_collection(collection_name)
            collection.data.delete_many(where=None)  # Delete all

            logger.warning(
                f"Deleted {initial_count} objects from collection '{collection_name}'"
            )
            return initial_count

        except Exception as e:
            error_msg = f"Failed to delete objects: {e}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

    def reset_collection(
        self,
        collection_name: Optional[str] = None,
        confirm: bool = False
    ) -> bool:
        """Reset a collection by deleting and recreating it.

        This completely removes the collection and recreates it with
        the schema from the configuration file.

        Args:
            collection_name: Collection name. Uses default if None.
            confirm: Must be True to proceed (safety check).

        Returns:
            True if reset successful.

        Raises:
            DatabaseError: If reset fails.

        Example:
            >>> manager.reset_collection(confirm=True)
            >>> # Collection is now empty with fresh schema
        """
        if not confirm:
            raise DatabaseError(
                "Must set confirm=True to reset collection. "
                "This will delete all data."
            )

        collection_name = collection_name or self.collection_name

        try:
            # Delete collection if it exists
            if self.collection_exists(collection_name):
                self.delete_schema(collection_name, confirm=True)
                logger.info(f"Deleted existing collection: {collection_name}")

            # Recreate schema
            self.create_schema(skip_if_exists=False)
            logger.info(f"Reset collection '{collection_name}' successfully")

            return True

        except Exception as e:
            error_msg = f"Failed to reset collection: {e}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

    def list_all_collections(self) -> List[str]:
        """List all collections in the Weaviate instance.

        Returns:
            List of collection names.

        Example:
            >>> collections = manager.list_all_collections()
            >>> print(f"Found {len(collections)} collections")
        """
        try:
            client = self.connector.get_client()
            collections = client.collections.list_all()
            collection_names = list(collections.keys())

            logger.info(f"Found {len(collection_names)} collections")
            return collection_names

        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"WeaviateManager("
            f"collection={self.collection_name}, "
            f"connected={self.connector.is_connected()}"
            f")"
        )


def get_manager(
    connector: Optional[WeaviateConnector] = None,
    settings: Optional[Settings] = None
) -> WeaviateManager:
    """Get a WeaviateManager instance.

    Convenience function to create a manager with default connector and settings.

    Args:
        connector: WeaviateConnector instance. If None, creates new one.
        settings: Settings instance. If None, loads default.

    Returns:
        WeaviateManager instance.

    Example:
        >>> from src.database.manager import get_manager
        >>> manager = get_manager()
        >>> manager.create_schema()
    """
    if settings is None:
        from config.settings import get_settings
        settings = get_settings()

    if connector is None:
        from src.database.connector import get_connector
        connector = get_connector(settings)

    return WeaviateManager(connector, settings)
