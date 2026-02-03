"""Custom exception hierarchy for the RAG system.

This module defines custom exceptions for different components of the system,
enabling precise error handling and debugging.

All exceptions inherit from RAGSystemError base class.
"""


class RAGSystemError(Exception):
    """Base exception for all RAG system errors.

    All custom exceptions in the system inherit from this base class,
    making it easy to catch any system-specific error.
    """
    pass


# Module 1: Knowledge Base Management Exceptions

class DocumentIngestionError(RAGSystemError):
    """Raised when document ingestion fails.

    Examples:
        - PDF parsing error
        - File not found
        - Unsupported file format
    """
    pass


class DocumentLoadError(RAGSystemError):
    """Raised when document loading fails.

    Examples:
        - File not found
        - Unsupported file format
        - PDF extraction error
        - Text encoding error
    """
    pass


class ChunkingError(RAGSystemError):
    """Raised when document chunking fails.

    Examples:
        - Semantic chunking algorithm failure
        - Invalid chunk size configuration
        - Text processing error
    """
    pass


class EmbeddingError(RAGSystemError):
    """Raised when embedding generation fails.

    Examples:
        - BGE-M3 model loading failure
        - Embedding computation error
        - Invalid input for embedding
    """
    pass


class IndexingError(RAGSystemError):
    """Raised when Weaviate indexing fails.

    Examples:
        - Connection to Weaviate failed
        - Schema mismatch
        - Batch upload error
    """
    pass


# Module 2: Retrieval Engine Exceptions

class QueryProcessingError(RAGSystemError):
    """Raised when query processing fails.

    Examples:
        - Thai text normalization error
        - HyDE generation failure
        - Category inference error
    """
    pass


class SearchError(RAGSystemError):
    """Raised when hybrid search operation fails.

    Examples:
        - Weaviate query execution error
        - Invalid search parameters
        - Connection timeout
    """
    pass


class RetrievalError(RAGSystemError):
    """Raised when the overall retrieval pipeline fails.

    This is a high-level exception that may wrap other retrieval-related
    exceptions (QueryProcessingError, SearchError, etc.).
    """
    pass


# Module 3: Response Generation Exceptions

class GenerationError(RAGSystemError):
    """Raised when response generation pipeline fails.

    This is a high-level exception that may wrap other generation-related
    exceptions (LLMAPIError, CitationError, etc.).
    """
    pass


class LLMAPIError(RAGSystemError):
    """Raised when LLM API call fails.

    Examples:
        - DeepSeek API timeout
        - Invalid API key
        - Rate limit exceeded
        - Network error
    """
    pass


class CitationError(RAGSystemError):
    """Raised when citation validation or enforcement fails.

    Examples:
        - Citations missing in response
        - Invalid citation format
        - Citation extraction error
    """
    pass


# Module 4: Evaluation Exceptions

class EvaluationError(RAGSystemError):
    """Raised when evaluation pipeline fails.

    Examples:
        - Metric computation error
        - Golden dataset loading failure
        - Evaluation results saving error
    """
    pass


# Database Exceptions

class DatabaseError(RAGSystemError):
    """Raised when database operations fail.

    Examples:
        - Weaviate connection error
        - Schema creation failure
        - Query execution error
    """
    pass


# Configuration Exceptions

class ConfigurationError(RAGSystemError):
    """Raised when configuration loading or validation fails.

    Examples:
        - Missing required environment variable
        - Invalid YAML configuration
        - Model config validation error
    """
    pass
