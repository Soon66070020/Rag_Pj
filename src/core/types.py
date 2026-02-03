"""Core type definitions for the RAG system.

This module defines dataclasses and type aliases used throughout
the application to ensure type safety and clear data contracts.

All dataclasses use Google-style docstrings as per project requirements.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class Document:
    """Represents a single chunk of knowledge base content.

    Attributes:
        id: Unique identifier for the chunk.
        content: Text content of the chunk.
        category: Category classification (e.g., "Post-op Care", "Medication").
        source_file: Original filename.
        page_number: Page number in source document.
        dense_vector: Dense embedding vector (from BGE-M3).
        sparse_vector: Learned sparse vector (from BGE-M3).
        metadata: Additional metadata dictionary.
    """
    id: str
    content: str
    category: str
    source_file: str
    page_number: int
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Query:
    """Represents a user query with preprocessing results.

    Attributes:
        original_text: Raw user input (Thai or mixed Thai-English).
        processed_text: Cleaned and normalized text.
        hyde_expansion: HyDE-generated hypothetical document.
        inferred_category: Predicted category for filtering.
        dense_vector: Dense embedding of query.
        sparse_vector: Sparse representation of query.
        timestamp: When query was received.
    """
    original_text: str
    processed_text: str = ""
    hyde_expansion: str = ""
    inferred_category: Optional[str] = None
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[Dict[str, float]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SearchResult:
    """Represents a single retrieved document with relevance score.

    Attributes:
        document: The retrieved document chunk.
        score: Relevance score (0.0 to 1.0).
        rank: Position in result set.
        score_breakdown: Details of scoring (dense, sparse, combined).
    """
    document: Document
    score: float
    rank: int
    score_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Complete retrieval pipeline output.

    Attributes:
        query: Original query object.
        candidates: Initial hybrid search results (Top-20).
        reranked_results: Final reranked results (Top-5).
        metadata_filters: Applied metadata filters.
        retrieval_time_ms: Total retrieval time in milliseconds.
    """
    query: Query
    candidates: List[SearchResult]
    reranked_results: List[SearchResult]
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    retrieval_time_ms: float = 0.0


@dataclass
class GeneratedResponse:
    """Complete response generation output.

    Attributes:
        query_text: Original user question.
        answer: Generated answer text (in Thai).
        citations: List of citation strings.
        context_used: Documents used for generation.
        generation_time_ms: Time taken to generate in milliseconds.
        model_name: LLM model used.
    """
    query_text: str
    answer: str
    citations: List[str]
    context_used: List[Document]
    generation_time_ms: float = 0.0
    model_name: str = "deepseek-chat"
