"""Unit tests for database module (connector and manager).

These tests verify the database layer functionality using mocks
to avoid requiring a real Weaviate instance during testing.

Run tests with:
    pytest tests/test_database.py -v
    pytest tests/test_database.py -v --cov=src.database
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from pathlib import Path
import json

from src.database.connector import WeaviateConnector, get_connector
from src.database.manager import WeaviateManager, get_manager
from src.core.exceptions import DatabaseError
from config.settings import Settings


# Fixtures

@pytest.fixture
def mock_settings():
    """Create a mock Settings instance."""
    settings = Mock(spec=Settings)
    settings.weaviate_url = "http://localhost:8080"
    settings.weaviate_api_key = None
    settings.config_dir = Path(__file__).parent.parent / "config"
    settings.model_config = {
        "weaviate": {
            "collection_name": "TestCollection"
        }
    }
    return settings


@pytest.fixture
def mock_weaviate_client():
    """Create a mock Weaviate client."""
    client = MagicMock()
    client.is_ready.return_value = True
    client.get_meta.return_value = {"version": "1.23.0"}
    client.collections = MagicMock()
    return client


@pytest.fixture
def mock_connector(mock_settings, mock_weaviate_client):
    """Create a mock WeaviateConnector."""
    with patch('src.database.connector.weaviate.connect_to_custom') as mock_connect:
        mock_connect.return_value = mock_weaviate_client
        connector = WeaviateConnector(mock_settings)
        return connector


@pytest.fixture
def mock_manager(mock_connector, mock_settings):
    """Create a WeaviateManager with mocked connector."""
    return WeaviateManager(mock_connector, mock_settings, "TestCollection")


# WeaviateConnector Tests

class TestWeaviateConnector:
    """Test suite for WeaviateConnector class."""

    def test_connector_initialization(self, mock_settings, mock_weaviate_client):
        """Test connector initializes and connects successfully."""
        with patch('src.database.connector.weaviate.connect_to_custom') as mock_connect:
            mock_connect.return_value = mock_weaviate_client

            connector = WeaviateConnector(mock_settings)

            assert connector.settings == mock_settings
            assert connector.is_connected() is True
            mock_connect.assert_called_once()

    def test_connector_with_api_key(self, mock_settings, mock_weaviate_client):
        """Test connector handles API key authentication."""
        mock_settings.weaviate_api_key = "test-api-key"

        with patch('src.database.connector.weaviate.connect_to_custom') as mock_connect:
            with patch('src.database.connector.AuthApiKey') as mock_auth:
                mock_connect.return_value = mock_weaviate_client

                connector = WeaviateConnector(mock_settings)

                assert connector.is_connected() is True
                mock_auth.assert_called_once_with(api_key="test-api-key")

    def test_connector_connection_failure(self, mock_settings):
        """Test connector raises DatabaseError on connection failure."""
        with patch('src.database.connector.weaviate.connect_to_custom') as mock_connect:
            mock_connect.side_effect = Exception("Connection refused")

            with pytest.raises(DatabaseError) as exc_info:
                WeaviateConnector(mock_settings)

            assert "Failed to connect" in str(exc_info.value)

    def test_health_check_success(self, mock_connector):
        """Test health check returns True when healthy."""
        assert mock_connector.health_check() is True

    def test_health_check_failure(self, mock_connector):
        """Test health check returns False when unhealthy."""
        mock_connector.client.is_ready.return_value = False
        assert mock_connector.health_check() is False

    def test_health_check_no_client(self, mock_connector):
        """Test health check returns False when no client."""
        mock_connector.client = None
        assert mock_connector.health_check() is False

    def test_disconnect(self, mock_connector):
        """Test disconnect closes client and updates status."""
        mock_connector.disconnect()

        assert mock_connector.is_connected() is False
        assert mock_connector.client is None

    def test_ensure_connection_when_disconnected(self, mock_connector, mock_weaviate_client):
        """Test ensure_connection reconnects when disconnected."""
        # Simulate disconnection
        mock_connector._connected = False
        mock_connector.client = None

        with patch('src.database.connector.weaviate.connect_to_custom') as mock_connect:
            mock_connect.return_value = mock_weaviate_client
            mock_connector.ensure_connection()

            assert mock_connector.is_connected() is True

    def test_get_client_success(self, mock_connector):
        """Test get_client returns active client."""
        client = mock_connector.get_client()
        assert client is not None

    def test_get_server_version(self, mock_connector):
        """Test get_server_version retrieves version."""
        version = mock_connector.get_server_version()
        assert version == "1.23.0"

    def test_context_manager(self, mock_settings, mock_weaviate_client):
        """Test connector works as context manager."""
        with patch('src.database.connector.weaviate.connect_to_custom') as mock_connect:
            mock_connect.return_value = mock_weaviate_client

            with WeaviateConnector(mock_settings) as connector:
                assert connector.is_connected() is True

            # After exiting context, should be disconnected
            # (but in our mock, we'd need to check disconnect was called)

    def test_get_connector_singleton(self, mock_settings, mock_weaviate_client):
        """Test get_connector returns singleton instance."""
        with patch('src.database.connector.weaviate.connect_to_custom') as mock_connect:
            mock_connect.return_value = mock_weaviate_client

            # Clear singleton
            import src.database.connector
            src.database.connector._connector_instance = None

            connector1 = get_connector(mock_settings)
            connector2 = get_connector(mock_settings)

            assert connector1 is connector2

    def test_repr(self, mock_connector):
        """Test string representation."""
        repr_str = repr(mock_connector)
        assert "WeaviateConnector" in repr_str
        assert "connected" in repr_str


# WeaviateManager Tests

class TestWeaviateManager:
    """Test suite for WeaviateManager class."""

    def test_manager_initialization(self, mock_connector, mock_settings):
        """Test manager initializes correctly."""
        manager = WeaviateManager(mock_connector, mock_settings, "TestCollection")

        assert manager.connector == mock_connector
        assert manager.settings == mock_settings
        assert manager.collection_name == "TestCollection"

    def test_load_schema_from_config(self, mock_manager):
        """Test loading schema from JSON config file."""
        # Create a temporary schema file
        schema_content = {
            "class": "TestCollection",
            "description": "Test collection",
            "properties": [
                {"name": "content", "dataType": ["text"]},
                {"name": "page_number", "dataType": ["int"]}
            ]
        }

        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(schema_content)

            with patch('pathlib.Path.exists', return_value=True):
                schema = mock_manager.load_schema_from_config()

                assert schema["class"] == "TestCollection"
                assert len(schema["properties"]) == 2

    def test_load_schema_file_not_found(self, mock_manager):
        """Test loading schema raises error when file not found."""
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(DatabaseError) as exc_info:
                mock_manager.load_schema_from_config()

            assert "Schema file not found" in str(exc_info.value)

    def test_collection_exists_true(self, mock_manager):
        """Test collection_exists returns True when collection exists."""
        mock_manager.connector.client.collections.exists.return_value = True

        assert mock_manager.collection_exists() is True

    def test_collection_exists_false(self, mock_manager):
        """Test collection_exists returns False when collection doesn't exist."""
        mock_manager.connector.client.collections.exists.return_value = False

        assert mock_manager.collection_exists() is False

    def test_create_schema_success(self, mock_manager):
        """Test schema creation succeeds."""
        schema = {
            "class": "TestCollection",
            "description": "Test collection",
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Content field"
                }
            ],
            "vectorIndexConfig": {
                "distance": "cosine"
            }
        }

        mock_manager.connector.client.collections.exists.return_value = False
        mock_collection = MagicMock()
        mock_manager.connector.client.collections.create.return_value = mock_collection

        result = mock_manager.create_schema(schema, skip_if_exists=True)

        assert result is True
        mock_manager.connector.client.collections.create.assert_called_once()

    def test_create_schema_skip_if_exists(self, mock_manager):
        """Test schema creation skips when collection exists."""
        mock_manager.connector.client.collections.exists.return_value = True

        result = mock_manager.create_schema(skip_if_exists=True)

        assert result is False

    def test_delete_schema_success(self, mock_manager):
        """Test schema deletion succeeds with confirmation."""
        mock_manager.connector.client.collections.exists.return_value = True

        result = mock_manager.delete_schema(confirm=True)

        assert result is True
        mock_manager.connector.client.collections.delete.assert_called_once()

    def test_delete_schema_requires_confirmation(self, mock_manager):
        """Test schema deletion requires confirmation flag."""
        with pytest.raises(DatabaseError) as exc_info:
            mock_manager.delete_schema(confirm=False)

        assert "Must set confirm=True" in str(exc_info.value)

    def test_delete_schema_not_found(self, mock_manager):
        """Test schema deletion returns False when collection doesn't exist."""
        mock_manager.connector.client.collections.exists.return_value = False

        result = mock_manager.delete_schema(confirm=True)

        assert result is False

    def test_get_collection_success(self, mock_manager):
        """Test get_collection returns collection object."""
        mock_collection = MagicMock()
        mock_manager.connector.client.collections.exists.return_value = True
        mock_manager.connector.client.collections.get.return_value = mock_collection

        collection = mock_manager.get_collection()

        assert collection == mock_collection

    def test_get_collection_not_exists(self, mock_manager):
        """Test get_collection raises error when collection doesn't exist."""
        mock_manager.connector.client.collections.exists.return_value = False

        with pytest.raises(DatabaseError) as exc_info:
            mock_manager.get_collection()

        assert "does not exist" in str(exc_info.value)

    def test_get_collection_stats(self, mock_manager):
        """Test get_collection_stats returns statistics."""
        mock_collection = MagicMock()
        mock_aggregate = MagicMock()
        mock_aggregate.total_count = 42

        mock_manager.connector.client.collections.exists.return_value = True
        mock_manager.connector.client.collections.get.return_value = mock_collection
        mock_collection.aggregate.over_all.return_value = mock_aggregate

        stats = mock_manager.get_collection_stats()

        assert stats["object_count"] == 42
        assert stats["collection_name"] == "TestCollection"
        assert stats["exists"] is True

    def test_delete_all_objects_success(self, mock_manager):
        """Test delete_all_objects removes all objects."""
        mock_collection = MagicMock()
        mock_aggregate = MagicMock()
        mock_aggregate.total_count = 10

        mock_manager.connector.client.collections.exists.return_value = True
        mock_manager.connector.client.collections.get.return_value = mock_collection
        mock_collection.aggregate.over_all.return_value = mock_aggregate

        count = mock_manager.delete_all_objects(confirm=True)

        assert count == 10
        mock_collection.data.delete_many.assert_called_once()

    def test_delete_all_objects_requires_confirmation(self, mock_manager):
        """Test delete_all_objects requires confirmation."""
        with pytest.raises(DatabaseError) as exc_info:
            mock_manager.delete_all_objects(confirm=False)

        assert "Must set confirm=True" in str(exc_info.value)

    def test_reset_collection(self, mock_manager):
        """Test reset_collection deletes and recreates collection."""
        schema = {
            "class": "TestCollection",
            "properties": [],
            "vectorIndexConfig": {"distance": "cosine"}
        }

        mock_manager.connector.client.collections.exists.return_value = True

        with patch.object(mock_manager, 'load_schema_from_config', return_value=schema):
            with patch.object(mock_manager, 'delete_schema', return_value=True):
                with patch.object(mock_manager, 'create_schema', return_value=True):
                    result = mock_manager.reset_collection(confirm=True)

                    assert result is True

    def test_list_all_collections(self, mock_manager):
        """Test list_all_collections returns collection names."""
        mock_collections = {
            "Collection1": MagicMock(),
            "Collection2": MagicMock()
        }
        mock_manager.connector.client.collections.list_all.return_value = mock_collections

        collections = mock_manager.list_all_collections()

        assert len(collections) == 2
        assert "Collection1" in collections
        assert "Collection2" in collections

    def test_get_manager_convenience_function(self, mock_connector, mock_settings):
        """Test get_manager convenience function."""
        manager = get_manager(mock_connector, mock_settings)

        assert isinstance(manager, WeaviateManager)
        assert manager.connector == mock_connector

    def test_repr(self, mock_manager):
        """Test string representation."""
        repr_str = repr(mock_manager)
        assert "WeaviateManager" in repr_str
        assert "TestCollection" in repr_str


# Integration-style tests (would require real Weaviate in integration tests)

class TestDatabaseIntegration:
    """Integration-style tests (mocked for unit testing)."""

    def test_full_workflow(self, mock_connector, mock_settings):
        """Test complete workflow: connect -> create schema -> get stats."""
        manager = WeaviateManager(mock_connector, mock_settings)

        # Mock schema creation
        schema = {
            "class": "TestCollection",
            "properties": [{"name": "content", "dataType": ["text"]}],
            "vectorIndexConfig": {"distance": "cosine"}
        }

        mock_connector.client.collections.exists.return_value = False
        mock_connector.client.collections.create.return_value = MagicMock()

        # Create schema
        created = manager.create_schema(schema)
        assert created is True

        # Verify it exists
        mock_connector.client.collections.exists.return_value = True
        assert manager.collection_exists() is True

    def test_error_handling_chain(self, mock_manager):
        """Test proper error propagation through the stack."""
        mock_manager.connector.client.collections.get.side_effect = Exception("Network error")

        with pytest.raises(DatabaseError) as exc_info:
            mock_manager.get_collection()

        assert "Failed to get collection" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.database"])
