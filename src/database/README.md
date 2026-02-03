# Database Module Documentation

The database module provides a robust interface to Weaviate vector database for the RAG system, with comprehensive connection management, schema handling, and data operations.

## Architecture

```
src/database/
├── __init__.py
├── connector.py        # Weaviate connection management
├── manager.py          # Schema and data operations
└── README.md          # This file
```

## Components

### 1. WeaviateConnector (connector.py)

Manages connections to Weaviate with:
- Singleton pattern for connection pooling
- Automatic reconnection on failure
- Health checks and server version info
- Context manager support
- Authentication handling

#### Key Features:
- ✅ Automatic connection management
- ✅ Health monitoring
- ✅ Connection pooling via singleton
- ✅ Proper error handling with `DatabaseError`
- ✅ Support for API key authentication

#### Usage:

```python
from src.database.connector import get_connector
from config.settings import get_settings

# Get singleton connector
settings = get_settings()
connector = get_connector(settings)

# Check connection
if connector.is_connected():
    print("Connected!")

# Get client for operations
client = connector.get_client()

# Health check
if connector.health_check():
    print("Weaviate is healthy")

# Get server version
version = connector.get_server_version()
print(f"Weaviate v{version}")

# Context manager usage
with get_connector() as conn:
    client = conn.get_client()
    # Do operations...
```

### 2. WeaviateManager (manager.py)

High-level interface for schema and data operations:
- Schema creation from JSON config
- Collection management (create, delete, reset)
- Data operations (bulk delete, statistics)
- Collection inspection

#### Key Features:
- ✅ Load schema from `config/weaviate_schema.json`
- ✅ Create/delete collections
- ✅ Reset collections (delete + recreate)
- ✅ Get collection statistics
- ✅ Bulk data operations
- ✅ Safety confirmations for destructive operations

#### Usage:

```python
from src.database.manager import get_manager

# Get manager
manager = get_manager()

# Check if collection exists
if manager.collection_exists():
    print("Collection exists")

# Create schema
manager.create_schema()  # Loads from config/weaviate_schema.json

# Get statistics
stats = manager.get_collection_stats()
print(f"Objects: {stats['object_count']}")

# Get collection for data operations
collection = manager.get_collection()
results = collection.query.fetch_objects(limit=10)

# List all collections
collections = manager.list_all_collections()
print(f"Found {len(collections)} collections")

# Reset collection (requires confirmation)
manager.reset_collection(confirm=True)
```

## Configuration

### Environment Variables (.env)

```env
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your_api_key_here  # Optional
```

### Schema Definition (config/weaviate_schema.json)

```json
{
  "class": "MedicalGuideline",
  "description": "Post-oral surgery dental guidelines",
  "vectorizer": "none",
  "properties": [
    {
      "name": "content",
      "dataType": ["text"],
      "description": "Document content"
    },
    {
      "name": "category",
      "dataType": ["text"],
      "indexFilterable": true
    }
  ],
  "vectorIndexConfig": {
    "distance": "cosine"
  }
}
```

## Error Handling

All database operations raise `DatabaseError` from `src.core.exceptions` on failure:

```python
from src.core.exceptions import DatabaseError

try:
    connector = get_connector()
    manager = get_manager(connector)
    manager.create_schema()
except DatabaseError as e:
    logger.error(f"Database operation failed: {e}")
```

## Testing

### Unit Tests

Run unit tests with pytest:

```bash
# Run all database tests
pytest tests/test_database.py -v

# Run with coverage
pytest tests/test_database.py -v --cov=src.database

# Run specific test class
pytest tests/test_database.py::TestWeaviateConnector -v
```

### Manual Testing

Use the setup script:

```bash
# Check database status
python scripts/setup_database.py --check

# Create schema
python scripts/setup_database.py

# Reset collection (deletes all data!)
python scripts/setup_database.py --reset

# Force recreate
python scripts/setup_database.py --force
```

## Common Operations

### Initialize Database

```python
from src.database.connector import get_connector
from src.database.manager import get_manager

# Connect
connector = get_connector()

# Create manager
manager = get_manager(connector)

# Create schema if not exists
if not manager.collection_exists():
    manager.create_schema()
    print("Schema created")
```

### Check Database Health

```python
connector = get_connector()

# Connection health
if connector.health_check():
    print("Connection healthy")

# Server info
version = connector.get_server_version()
print(f"Weaviate v{version}")

# Collection stats
manager = get_manager(connector)
stats = manager.get_collection_stats()
print(f"Objects: {stats['object_count']}")
```

### Delete All Data (Keep Schema)

```python
manager = get_manager()

# Delete all objects (requires confirmation)
count = manager.delete_all_objects(confirm=True)
print(f"Deleted {count} objects")
```

### Complete Reset

```python
manager = get_manager()

# Delete collection and recreate (requires confirmation)
manager.reset_collection(confirm=True)
print("Collection reset")
```

## Design Patterns

### Singleton Pattern

The connector uses singleton pattern for connection pooling:

```python
# These will return the same instance
connector1 = get_connector()
connector2 = get_connector()
assert connector1 is connector2
```

### Context Manager

Connector supports context manager for automatic cleanup:

```python
with get_connector() as conn:
    client = conn.get_client()
    # Operations...
# Connection automatically managed
```

### Dependency Injection

Manager accepts connector and settings via dependency injection:

```python
# Custom configuration
custom_settings = Settings()
custom_connector = WeaviateConnector(custom_settings)
manager = WeaviateManager(custom_connector, custom_settings)
```

## Safety Features

### Confirmation Required for Destructive Operations

All destructive operations require explicit confirmation:

```python
# These will raise DatabaseError without confirm=True
manager.delete_schema(confirm=True)
manager.delete_all_objects(confirm=True)
manager.reset_collection(confirm=True)
```

### Automatic Reconnection

Connection is automatically restored if lost:

```python
connector = get_connector()
# Connection lost...
client = connector.get_client()  # Automatically reconnects
```

### Health Monitoring

Regular health checks prevent operations on dead connections:

```python
connector.ensure_connection()  # Checks and reconnects if needed
```

## Logging

All operations are logged for debugging:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Database operations will log:
# - Connection attempts
# - Schema operations
# - Health checks
# - Errors with stack traces
```

## Best Practices

1. **Always use get_connector() and get_manager()**
   - Ensures singleton pattern
   - Proper connection pooling

2. **Check collection exists before operations**
   ```python
   if not manager.collection_exists():
       manager.create_schema()
   ```

3. **Use confirmation flags for destructive operations**
   ```python
   manager.delete_all_objects(confirm=True)
   ```

4. **Handle DatabaseError exceptions**
   ```python
   try:
       manager.create_schema()
   except DatabaseError as e:
       logger.error(f"Failed: {e}")
   ```

5. **Use context manager for temporary connections**
   ```python
   with get_connector(force_new=True) as conn:
       # Isolated connection
       pass
   ```

## Troubleshooting

### Connection Refused

```python
# Check Weaviate is running
docker ps | grep weaviate

# Verify URL in .env
echo $WEAVIATE_URL
```

### Schema Creation Failed

```python
# Check schema JSON syntax
cat config/weaviate_schema.json | python -m json.tool

# Check collection doesn't exist with wrong name
manager.list_all_collections()
```

### Health Check Fails

```python
# Check Weaviate logs
docker logs weaviate

# Test connection manually
curl http://localhost:8080/v1/.well-known/ready
```

## API Reference

See docstrings in source files for complete API documentation:
- [connector.py](connector.py) - Connection management
- [manager.py](manager.py) - Schema and data operations
