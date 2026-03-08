# =============================================================================
# Dockerfile for Dental Post-Op RAG API
# Multi-stage build: builder (compile deps) -> runtime (slim)
# CPU-only torch — auto-detects CUDA at runtime if available
# =============================================================================

# Stage 1: Builder - install Python dependencies
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

# Install CPU-only torch first (smaller), then remaining deps
RUN pip install --no-cache-dir --prefix=/install \
    torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime - slim image with only what we need
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application code
COPY api/ ./api/
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY examples/ ./examples/

# Copy entrypoint
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Create directories for volumes
RUN mkdir -p /app/models_cache /app/pythainlp_data /app/data/raw /app/logs

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/models_cache \
    TRANSFORMERS_CACHE=/app/models_cache \
    TORCH_HOME=/app/models_cache \
    PYTHAINLP_DATA_DIR=/app/pythainlp_data \
    WEAVIATE_URL=http://weaviate:8080 \
    LOG_LEVEL=INFO \
    LOG_DIR=/app/logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["api"]
