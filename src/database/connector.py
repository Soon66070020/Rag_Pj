"""Weaviate database connector module.

This module provides a connection interface to Weaviate vector database,
handling authentication, connection pooling, and health checks.

The connector follows the singleton pattern to ensure a single connection
instance is reused throughout the application lifecycle.
"""

import logging
from typing import Optional
import weaviate
from weaviate.client import WeaviateClient
from weaviate.auth import AuthApiKey

from config.settings import Settings
from src.core.exceptions import DatabaseError


logger = logging.getLogger(__name__)


class WeaviateConnector:
    """Weaviate database connector with connection pooling and health checks.

    This class manages the connection to Weaviate, providing:
    - Automatic reconnection on failure
    - Health check capabilities
    - Proper resource cleanup
    - Authentication handling

    Attributes:
        settings: Configuration settings instance.
        client: Weaviate client instance.
        _connected: Connection status flag.
    """

    def __init__(self, settings: Settings):
        """Initialize the Weaviate connector.

        Args:
            settings: Settings instance containing Weaviate configuration.

        Raises:
            DatabaseError: If initial connection fails.
        """
        self.settings = settings
        self.client: Optional[WeaviateClient] = None
        self._connected = False

        # Attempt initial connection
        self.connect()

    def connect(self) -> WeaviateClient:
        """Establish connection to Weaviate database.

        Returns:
            Connected Weaviate client instance.

        Raises:
            DatabaseError: If connection fails.

        Example:
            >>> connector = WeaviateConnector(settings)
            >>> client = connector.connect()
            >>> print(connector.is_connected())
            True
        """
        try:
            logger.info(f"Connecting to Weaviate at {self.settings.weaviate_url}")

            # Parse URL to extract host, port, and protocol
            from urllib.parse import urlparse
            parsed_url = urlparse(self.settings.weaviate_url)

            http_host = parsed_url.hostname or "localhost"
            http_port = parsed_url.port or (443 if parsed_url.scheme == "https" else 8080)
            http_secure = parsed_url.scheme == "https"

            # Build connection parameters for Weaviate v4 API
            connection_params = {
                "http_host": http_host,
                "http_port": http_port,
                "http_secure": http_secure,
                "grpc_host": http_host,
                "grpc_port": 50051,  # Default gRPC port
                "grpc_secure": http_secure,
            }

            # Add authentication if API key is provided
            if self.settings.weaviate_api_key:
                connection_params["auth_credentials"] = AuthApiKey(
                    api_key=self.settings.weaviate_api_key
                )
                logger.debug("Using API key authentication")

            # Create Weaviate client (v4 API)
            self.client = weaviate.connect_to_custom(**connection_params)

            # Verify connection with health check
            if not self.health_check():
                raise DatabaseError("Health check failed after connection")

            self._connected = True
            logger.info("Successfully connected to Weaviate")

            return self.client

        except Exception as e:
            self._connected = False
            error_msg = f"Failed to connect to Weaviate at {self.settings.weaviate_url}: {e}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

    def disconnect(self):
        """Close the connection to Weaviate database.

        Properly closes the client connection and releases resources.

        Example:
            >>> connector.disconnect()
            >>> print(connector.is_connected())
            False
        """
        if self.client:
            try:
                self.client.close()
                logger.info("Disconnected from Weaviate")
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self._connected = False
                self.client = None

    def health_check(self) -> bool:
        """Perform health check on Weaviate connection.

        Verifies that the connection is alive and Weaviate is responding.

        Returns:
            True if connection is healthy, False otherwise.

        Example:
            >>> if connector.health_check():
            ...     print("Connection is healthy")
        """
        if not self.client:
            return False

        try:
            # Check if client is ready
            is_ready = self.client.is_ready()

            if is_ready:
                logger.debug("Weaviate health check: OK")
            else:
                logger.warning("Weaviate health check: NOT READY")

            return is_ready

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def is_connected(self) -> bool:
        """Check if currently connected to Weaviate.

        Returns:
            True if connected, False otherwise.

        Example:
            >>> if not connector.is_connected():
            ...     connector.connect()
        """
        return self._connected and self.client is not None

    def ensure_connection(self):
        """Ensure connection is active, reconnect if necessary.

        Automatically attempts to reconnect if connection is lost.

        Raises:
            DatabaseError: If reconnection fails.

        Example:
            >>> connector.ensure_connection()
            >>> # Now safe to use client
            >>> connector.client.collections.list_all()
        """
        if not self.is_connected() or not self.health_check():
            logger.warning("Connection lost, attempting to reconnect...")
            self.connect()

    def get_client(self) -> WeaviateClient:
        """Get the Weaviate client instance, ensuring connection is active.

        Returns:
            Active Weaviate client instance.

        Raises:
            DatabaseError: If not connected and reconnection fails.

        Example:
            >>> client = connector.get_client()
            >>> collections = client.collections.list_all()
        """
        self.ensure_connection()

        if not self.client:
            raise DatabaseError("Failed to obtain Weaviate client")

        return self.client

    def get_server_version(self) -> str:
        """Get Weaviate server version information.

        Returns:
            Version string of the connected Weaviate instance.

        Raises:
            DatabaseError: If unable to retrieve version.

        Example:
            >>> version = connector.get_server_version()
            >>> print(f"Weaviate version: {version}")
        """
        try:
            client = self.get_client()
            meta = client.get_meta()
            version = meta.get("version", "unknown")
            logger.info(f"Weaviate version: {version}")
            return version

        except Exception as e:
            error_msg = f"Failed to get server version: {e}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

    def __enter__(self):
        """Context manager entry."""
        self.ensure_connection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def __repr__(self) -> str:
        """String representation of connector."""
        status = "connected" if self._connected else "disconnected"
        return f"WeaviateConnector(url={self.settings.weaviate_url}, status={status})"


# Singleton instance management
_connector_instance: Optional[WeaviateConnector] = None


def get_connector(settings: Optional[Settings] = None, force_new: bool = False) -> WeaviateConnector:
    """Get or create the global WeaviateConnector instance.

    Implements singleton pattern for connection pooling.

    Args:
        settings: Settings instance. If None, loads from get_settings().
        force_new: If True, creates a new connector instance.

    Returns:
        WeaviateConnector instance.

    Example:
        >>> from src.database.connector import get_connector
        >>> connector = get_connector()
        >>> client = connector.get_client()
    """
    global _connector_instance

    if _connector_instance is None or force_new:
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()

        _connector_instance = WeaviateConnector(settings)

    return _connector_instance
