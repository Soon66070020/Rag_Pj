"""Retrieval module for RAG pipeline.

This module provides the complete retrieval pipeline for the RAG system:
1. Query Processing: Thai text normalization, embedding generation
2. Hybrid Search: Alpha-blended dense + sparse search in Weaviate
3. Reranking: Cross-encoder refinement with BGE-Reranker-v2-m3

Optimized for low latency and high relevance (Recall@K).

Main Classes:
    - QueryProcessor: Process and embed user queries
    - HybridSearcher: Alpha-blended hybrid search in Weaviate
    - BGEReranker: Cross-encoder reranking
    - RetrievalService: Complete retrieval pipeline facade

Main Function:
    - retrieve(query_text: str) -> RetrievalResult

Example:
    >>> from src.retrieval import retrieve
    >>> result = retrieve("ฉันควรกินยาแก้ปวดไหม")
    >>> for doc in result.reranked_results:
    ...     print(f"{doc.rank}. {doc.document.content[:50]}...")
"""

from src.retrieval.query_processor import (
    QueryProcessor,
    process_query
)

from src.retrieval.hybrid_search import (
    HybridSearcher,
    hybrid_search
)

from src.retrieval.reranker import (
    BGEReranker,
    rerank_results
)

from src.retrieval.service import (
    RetrievalService,
    retrieve
)


__all__ = [
    # Query Processing
    'QueryProcessor',
    'process_query',

    # Hybrid Search
    'HybridSearcher',
    'hybrid_search',

    # Reranking
    'BGEReranker',
    'rerank_results',

    # Service Facade
    'RetrievalService',
    'retrieve',
]
