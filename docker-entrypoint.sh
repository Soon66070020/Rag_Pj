#!/bin/bash
set -e

wait_for_weaviate() {
    local max_attempts=30
    local attempt=1
    local weaviate_health="${WEAVIATE_URL}/v1/.well-known/ready"

    echo "Waiting for Weaviate at ${WEAVIATE_URL}..."

    while [ $attempt -le $max_attempts ]; do
        if curl -sf "$weaviate_health" > /dev/null 2>&1; then
            echo "Weaviate is ready."
            return 0
        fi
        echo "Attempt $attempt/$max_attempts: Weaviate not ready, waiting 5s..."
        sleep 5
        attempt=$((attempt + 1))
    done

    echo "ERROR: Weaviate did not become ready within $((max_attempts * 5))s"
    return 1
}

case "${1}" in
    api)
        echo "Starting RAG API server..."
        wait_for_weaviate
        exec python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
        ;;
    setup)
        echo "Running database setup..."
        wait_for_weaviate

        python scripts/setup_database.py

        if [ -n "$(ls -A /app/data/raw/ 2>/dev/null)" ]; then
            echo "Found documents in data/raw/, starting ingestion..."
            python scripts/ingest_documents.py --directory /app/data/raw
        else
            echo "No documents found in data/raw/, skipping ingestion."
        fi

        echo "Setup complete."
        ;;
    ingest)
        echo "Running document ingestion..."
        wait_for_weaviate
        python scripts/ingest_documents.py --directory /app/data/raw
        ;;
    *)
        exec "$@"
        ;;
esac
