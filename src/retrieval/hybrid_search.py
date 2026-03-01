"""Hybrid search module using Weaviate vector database.

This module implements alpha-blended hybrid search combining:
- Dense vector search (semantic similarity)
- Sparse vector search (lexical matching)

Optimized for low latency with configurable alpha blending.
"""

import logging
from typing import List, Optional, Dict, Any

from src.core.types import Query, SearchResult, Document
from src.core.exceptions import SearchError
from src.database.manager import WeaviateManager


logger = logging.getLogger(__name__)


class HybridSearcher:
    """Hybrid search engine for Weaviate.

    Implements alpha-blended hybrid search:
    - combined_score = alpha * dense_score + (1-alpha) * sparse_score

    Where:
    - alpha = 1.0: Pure dense (semantic) search
    - alpha = 0.0: Pure sparse (keyword) search
    - alpha = 0.7 (default): Balanced hybrid search

    Attributes:
        db_manager: Weaviate database manager.
        alpha: Blending factor (0.0 to 1.0).
        top_k: Number of results to retrieve.

    Example:
        >>> searcher = HybridSearcher(db_manager, alpha=0.7)
        >>> results = searcher.search(query, top_k=20)
        >>> print(f"Found {len(results)} results")
    """

    def __init__(
        self,
        db_manager: WeaviateManager,
        alpha: float = 0.7,
        top_k: int = 20
    ):
        """Initialize hybrid searcher.

        Args:
            db_manager: Weaviate database manager.
            alpha: Blending factor (0.0 = sparse only, 1.0 = dense only).
            top_k: Number of results to retrieve.

        Raises:
            SearchError: If alpha is out of range.
        """
        if not 0.0 <= alpha <= 1.0:
            raise SearchError(f"Alpha must be between 0.0 and 1.0, got {alpha}")

        self.db_manager = db_manager
        self.alpha = alpha
        self.top_k = top_k

        logger.info(
            f"HybridSearcher initialized with alpha={alpha}, top_k={top_k}"
        )

    def search(
        self,
        query: Query,
        top_k: Optional[int] = None,
        category_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """Perform hybrid search in Weaviate.

        Args:
            query: Processed query with embeddings.
            top_k: Number of results to retrieve. Uses default if None.
            category_filter: Optional category to filter by.

        Returns:
            List of SearchResult objects sorted by relevance.

        Raises:
            SearchError: If search fails.

        Example:
            >>> results = searcher.search(query, top_k=20)
            >>> for result in results[:5]:
            ...     print(f"{result.rank}. {result.document.content[:50]}...")
        """
        if not query.dense_vector:
            raise SearchError("Query must have dense_vector for hybrid search")

        top_k = top_k or self.top_k

        try:
            logger.debug(
                f"Executing hybrid search: top_k={top_k}, "
                f"alpha={self.alpha}, category={category_filter}"
            )

            # Get Weaviate collection
            collection = self.db_manager.get_collection()

            # Build query filters
            filters = self._build_filters(category_filter)

            # Execute hybrid search
            # Weaviate v4 hybrid search with alpha parameter
            response = collection.query.hybrid(
                query=query.processed_text,  # Text query for BM25
                vector=query.dense_vector,   # Dense vector for semantic search
                alpha=self.alpha,            # Alpha blending
                limit=top_k,
                return_metadata=['score', 'distance'],
                filters=filters
            )

            # Parse results
            results = self._parse_results(response, query)

            logger.info(f"Retrieved {len(results)} results from hybrid search")
            return results

        except Exception as e:
            error_msg = f"Hybrid search failed: {e}"
            logger.error(error_msg)
            raise SearchError(error_msg) from e

    def _build_filters(self, category: Optional[str]) -> Optional[Any]:
        """Build Weaviate filters for category.

        Args:
            category: Category to filter by.

        Returns:
            Weaviate filter object or None.
        """
        if not category:
            return None

        try:
            from weaviate.classes.query import Filter

            # Filter by category field
            filters = Filter.by_property("category").equal(category)

            logger.debug(f"Applied category filter: {category}")
            return filters

        except Exception as e:
            logger.warning(f"Failed to build filters: {e}")
            return None

    def _parse_results(
        self,
        response: Any,
        query: Query
    ) -> List[SearchResult]:
        """Parse Weaviate response into SearchResult objects.

        Args:
            response: Weaviate query response.
            query: Original query.

        Returns:
            List of SearchResult objects.
        """
        results = []

        if not response or not response.objects:
            logger.warning("No results returned from Weaviate")
            return results

        for rank, obj in enumerate(response.objects, start=1):
            try:
                # Extract properties
                props = obj.properties

                # Create Document object
                document = Document(
                    id=str(obj.uuid),
                    content=props.get('content', ''),
                    category=props.get('category', 'Unknown'),
                    source_file=props.get('source_file', ''),
                    page_number=props.get('page_number', 0),
                    dense_vector=None,  # Don't return vectors to save memory
                    sparse_vector=None,
                    metadata={}
                )

                # Extract score
                metadata = obj.metadata
                score = metadata.score if metadata and hasattr(metadata, 'score') else 0.0

                # Create SearchResult
                result = SearchResult(
                    document=document,
                    score=float(score),
                    rank=rank,
                    score_breakdown={
                        'hybrid_score': float(score),
                        'alpha': self.alpha
                    }
                )

                results.append(result)

            except Exception as e:
                logger.warning(f"Failed to parse result at rank {rank}: {e}")
                continue

        return results

    def search_multi_category(
        self,
        query: Query,
        categories: List[str],
        top_k: Optional[int] = None,
        min_results: int = 5,
        score_boost: float = 0.1
    ) -> List[SearchResult]:
        """3-round fallback search using multi-category filtering.

        Round 1: Filter by categories array (contains_any).
        Round 2: If too few results, search without filter.
        Category-matched results get a score boost.

        Args:
            query: Processed query with embeddings.
            categories: List of L1 category names to filter by.
            top_k: Number of results to retrieve per round.
            min_results: Minimum results before falling back.
            score_boost: Score bonus for category-matched results.

        Returns:
            List of SearchResult objects, deduplicated and sorted.
        """
        top_k = top_k or self.top_k
        seen_ids: set = set()
        all_results: List[SearchResult] = []

        # Round 1: Search with category filter
        if categories:
            cat_filter = self._build_multi_category_filter(categories)
            if cat_filter:
                try:
                    collection = self.db_manager.get_collection()
                    response = collection.query.hybrid(
                        query=query.processed_text,
                        vector=query.dense_vector,
                        alpha=self.alpha,
                        limit=top_k,
                        return_metadata=['score', 'distance'],
                        filters=cat_filter,
                    )
                    round1 = self._parse_results(response, query)
                    for r in round1:
                        doc_id = r.document.id
                        if doc_id not in seen_ids:
                            r.score = r.score + score_boost
                            r.score_breakdown['category_boost'] = score_boost
                            all_results.append(r)
                            seen_ids.add(doc_id)

                    logger.info(
                        f"Round 1 (category filter {categories}): "
                        f"{len(round1)} results"
                    )
                except Exception as e:
                    logger.warning(f"Round 1 search failed: {e}")

        # Round 2: No filter (if not enough results)
        if len(all_results) < min_results:
            try:
                round2 = self.search(query, top_k=top_k, category_filter=None)
                for r in round2:
                    doc_id = r.document.id
                    if doc_id not in seen_ids:
                        all_results.append(r)
                        seen_ids.add(doc_id)

                logger.info(
                    f"Round 2 (no filter): {len(round2)} results, "
                    f"total unique: {len(all_results)}"
                )
            except Exception as e:
                logger.warning(f"Round 2 search failed: {e}")

        # Re-sort by score and re-rank
        all_results.sort(key=lambda r: r.score, reverse=True)
        for rank, result in enumerate(all_results, start=1):
            result.rank = rank

        return all_results

    def _build_multi_category_filter(self, categories: List[str]) -> Optional[Any]:
        """Build Weaviate filter for multi-category array field.

        Uses 'categories' text[] field with contains_any.

        Args:
            categories: List of L1 category names.

        Returns:
            Weaviate filter object or None.
        """
        if not categories:
            return None

        try:
            from weaviate.classes.query import Filter

            filters = Filter.by_property("categories").contains_any(categories)
            logger.debug(f"Applied multi-category filter: {categories}")
            return filters

        except Exception as e:
            logger.warning(f"Failed to build multi-category filter: {e}")
            return None

    def search_by_vector(
        self,
        vector: List[float],
        top_k: Optional[int] = None,
        category_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """Perform pure dense vector search.

        Args:
            vector: Dense query vector.
            top_k: Number of results to retrieve.
            category_filter: Optional category filter.

        Returns:
            List of SearchResult objects.

        Example:
            >>> results = searcher.search_by_vector(query.dense_vector, top_k=10)
        """
        top_k = top_k or self.top_k

        try:
            collection = self.db_manager.get_collection()
            filters = self._build_filters(category_filter)

            # Pure vector search (near_vector)
            response = collection.query.near_vector(
                near_vector=vector,
                limit=top_k,
                return_metadata=['distance'],
                filters=filters
            )

            # Create dummy query for parsing
            dummy_query = Query(original_text="", dense_vector=vector)
            results = self._parse_results(response, dummy_query)

            logger.info(f"Retrieved {len(results)} results from vector search")
            return results

        except Exception as e:
            error_msg = f"Vector search failed: {e}"
            logger.error(error_msg)
            raise SearchError(error_msg) from e

    def get_statistics(self) -> Dict[str, Any]:
        """Get search statistics and collection info.

        Returns:
            Dictionary with statistics.

        Example:
            >>> stats = searcher.get_statistics()
            >>> print(f"Total documents: {stats['total_documents']}")
        """
        try:
            stats = self.db_manager.get_collection_stats()

            return {
                'collection_name': stats.get('collection_name'),
                'total_documents': stats.get('object_count', 0),
                'alpha': self.alpha,
                'top_k': self.top_k
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}


def hybrid_search(
    query: Query,
    db_manager: WeaviateManager,
    alpha: float = 0.7,
    top_k: int = 20,
    category_filter: Optional[str] = None
) -> List[SearchResult]:
    """Convenience function for hybrid search.

    Args:
        query: Processed query with embeddings.
        db_manager: Weaviate database manager.
        alpha: Blending factor.
        top_k: Number of results.
        category_filter: Optional category filter.

    Returns:
        List of SearchResult objects.

    Example:
        >>> results = hybrid_search(query, db_manager, alpha=0.7, top_k=20)
    """
    searcher = HybridSearcher(db_manager, alpha=alpha, top_k=top_k)
    return searcher.search(query, category_filter=category_filter)
