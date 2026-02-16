"""Retrieval service facade.

This module provides a high-level interface to the retrieval pipeline,
combining query processing, hybrid search, and reranking into a single
function call.

Optimized for low latency and high relevance (Recall@K).
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict

from src.core.types import Query, SearchResult, RetrievalResult
from src.core.exceptions import RetrievalError
from src.retrieval.query_processor import QueryProcessor
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import BGEReranker
from src.database.manager import WeaviateManager
from src.generation.client import DeepSeekClient
from src.generation.prompt_builder import PromptBuilder
from config.settings import Settings


logger = logging.getLogger(__name__)


class RetrievalService:
    """Complete retrieval service for RAG pipeline.

    Orchestrates the full retrieval pipeline:
    1. Query Processing: Normalize text, generate embeddings, infer category
    2. Hybrid Search: Alpha-blended dense + sparse search in Weaviate
    3. Reranking: Cross-encoder refinement with BGE-Reranker

    Optimized for:
    - Low latency: Single-query embedding, efficient Weaviate queries
    - High relevance: Two-stage retrieval (broad recall → precise rerank)

    Attributes:
        query_processor: Query preprocessing engine.
        hybrid_searcher: Weaviate hybrid search engine.
        reranker: BGE cross-encoder reranker.
        use_category_filter: Whether to filter by inferred category.

    Example:
        >>> service = RetrievalService(settings)
        >>> result = service.retrieve("ฉันควรกินยาแก้ปวดไหม")
        >>> print(f"Top result: {result.reranked_results[0].document.content}")
    """

    def __init__(
        self,
        settings: Settings,
        db_manager: Optional[WeaviateManager] = None,
        use_category_filter: bool = True
    ):
        """Initialize retrieval service.

        Args:
            settings: Configuration settings.
            db_manager: Weaviate manager. Creates new if None.
            use_category_filter: Whether to use category-based filtering.
        """
        self.settings = settings
        self.use_category_filter = use_category_filter

        # Get configuration
        retrieval_config = settings.model_config.get('retrieval', {})
        reranker_config = settings.model_config.get('reranker', {})

        self.alpha = retrieval_config.get('alpha', 0.7)
        self.top_k = retrieval_config.get('top_k', 20)
        self.top_n = retrieval_config.get('top_n', 5)

        logger.info("Initializing RetrievalService...")

        # Initialize database manager
        if db_manager is None:
            from src.database.connector import get_connector
            connector = get_connector(settings)
            from src.database.manager import get_manager
            self.db_manager = get_manager(connector, settings)
        else:
            self.db_manager = db_manager

        # Initialize Q2D client (deepseek-chat for query expansion)
        q2d_client = None
        prompt_builder = None
        q2d_config = settings.model_config.get('q2d', {})
        if retrieval_config.get('use_q2d', False) and q2d_config:
            logger.info("Initializing Q2D client (deepseek-chat)...")
            q2d_client = DeepSeekClient(
                api_key=settings.deepseek_api_key,
                model_name=q2d_config.get('model_name', 'deepseek-chat'),
                api_base=q2d_config.get('api_base', 'https://api.deepseek.com'),
                temperature=q2d_config.get('temperature', 0.3),
                max_tokens=q2d_config.get('max_tokens', 6000),
                timeout=q2d_config.get('timeout', 120)
            )
            prompt_builder = PromptBuilder(settings.prompts)
            logger.info("Q2D client ready")

        # Initialize query processor (with shared embedding generator + Q2D)
        logger.info("Loading query processor...")
        self.query_processor = QueryProcessor(
            auto_infer_category=use_category_filter,
            q2d_client=q2d_client,
            prompt_builder=prompt_builder
        )

        # Initialize hybrid searcher
        logger.info("Initializing hybrid searcher...")
        self.hybrid_searcher = HybridSearcher(
            db_manager=self.db_manager,
            alpha=self.alpha,
            top_k=self.top_k
        )

        # Initialize reranker
        logger.info("Loading reranker model...")
        self.reranker = BGEReranker(
            model_name=reranker_config.get('model_name', 'BAAI/bge-reranker-v2-m3'),
            top_n=self.top_n,
            batch_size=reranker_config.get('batch_size', 32),
            device=reranker_config.get('device', 'cuda'),
            use_fp16=reranker_config.get('use_fp16', True)
        )

        logger.info(
            f"RetrievalService ready: alpha={self.alpha}, "
            f"top_k={self.top_k}, top_n={self.top_n}"
        )

    def retrieve(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        use_reranking: bool = True
    ) -> RetrievalResult:
        """Retrieve relevant documents for a query.

        Complete pipeline:
        1. Process query (normalize, embed, infer category)
        2. Hybrid search in Weaviate (top_k candidates)
        3. Rerank with BGE cross-encoder (top_n results)

        Args:
            query_text: User query in Thai or mixed Thai-English.
            top_k: Number of candidates from hybrid search. Uses default if None.
            top_n: Number of final results after reranking. Uses default if None.
            use_reranking: Whether to apply reranking (disable for faster retrieval).

        Returns:
            RetrievalResult with query, candidates, and reranked results.

        Raises:
            RetrievalError: If retrieval fails.

        Example:
            >>> result = service.retrieve("ฉันควรกินอาหารอะไรหลังผ่าตัด")
            >>> for i, res in enumerate(result.reranked_results, 1):
            ...     print(f"{i}. {res.document.content[:50]}... (score: {res.score:.3f})")
        """
        start_time = time.time()

        try:
            # Step 1: Process query
            logger.info(f"Processing query: '{query_text[:50]}...'")
            query = self.query_processor.process(query_text)

            # Determine category filter
            category_filter = None
            if self.use_category_filter and query.inferred_category:
                category_filter = query.inferred_category
                logger.debug(f"Using category filter: {category_filter}")

            # Step 2: Hybrid search (multi-query or single)
            k = top_k or self.top_k
            if query.multi_dense_vectors and len(query.multi_dense_vectors) > 1:
                logger.info(f"Executing multi-query search ({len(query.multi_dense_vectors)} variations)...")
                candidates = self._multi_query_search(query, k, category_filter)
            else:
                logger.info(f"Executing hybrid search (top_k={k})...")
                candidates = self.hybrid_searcher.search(
                    query=query,
                    top_k=k,
                    category_filter=category_filter
                )

            if not candidates:
                logger.warning("No candidates found in hybrid search")
                return RetrievalResult(
                    query=query,
                    candidates=[],
                    reranked_results=[],
                    metadata_filters={'category': category_filter} if category_filter else {},
                    retrieval_time_ms=(time.time() - start_time) * 1000
                )

            logger.info(f"Retrieved {len(candidates)} candidates")

            # Step 3: Reranking (optional)
            if use_reranking:
                logger.info(f"Reranking to top {top_n or self.top_n}...")
                reranked = self.reranker.rerank(
                    query=query,
                    candidates=candidates,
                    top_n=top_n
                )
            else:
                # Skip reranking, just take top_n from candidates
                n = top_n or self.top_n
                reranked = candidates[:n]
                for rank, result in enumerate(reranked, start=1):
                    result.rank = rank

            elapsed_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Retrieval complete: {len(reranked)} results in {elapsed_ms:.2f}ms"
            )

            # Build result
            result = RetrievalResult(
                query=query,
                candidates=candidates,
                reranked_results=reranked,
                metadata_filters={'category': category_filter} if category_filter else {},
                retrieval_time_ms=elapsed_ms
            )

            return result

        except Exception as e:
            error_msg = f"Retrieval failed for query '{query_text}': {e}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e

    def _multi_query_search(
        self,
        query: Query,
        top_k: int,
        category_filter: Optional[str]
    ) -> List[SearchResult]:
        """Search with multiple query variations and merge via RRF.

        Args:
            query: Query with multi_dense_vectors and multi_sparse_vectors.
            top_k: Number of candidates per variation.
            category_filter: Optional category filter.

        Returns:
            Merged and deduplicated search results.
        """
        per_query_k = max(top_k // 2, 10)

        def search_one(i: int) -> List[SearchResult]:
            temp_query = Query(
                original_text=query.original_text,
                processed_text=query.q2d_expansions[i] if i < len(query.q2d_expansions) else query.processed_text,
                dense_vector=query.multi_dense_vectors[i],
                sparse_vector=query.multi_sparse_vectors[i],
            )
            return self.hybrid_searcher.search(
                temp_query, top_k=per_query_k, category_filter=category_filter
            )

        num_variations = len(query.multi_dense_vectors)
        with ThreadPoolExecutor(max_workers=num_variations) as executor:
            futures = [executor.submit(search_one, i) for i in range(num_variations)]
            all_result_lists = [f.result() for f in futures]

        merged = self._rrf_merge(all_result_lists)

        total_raw = sum(len(r) for r in all_result_lists)
        logger.info(
            f"Multi-query search: {num_variations} variations, "
            f"{total_raw} raw -> {len(merged)} unique candidates"
        )

        return merged

    @staticmethod
    def _rrf_merge(
        result_lists: List[List[SearchResult]],
        k: int = 60
    ) -> List[SearchResult]:
        """Reciprocal Rank Fusion for combining multiple ranked lists.

        RRF_score(doc) = sum(1 / (k + rank_i)) for each list containing doc.

        Args:
            result_lists: List of ranked SearchResult lists.
            k: RRF constant (default 60).

        Returns:
            Merged results sorted by RRF score.
        """
        scores: Dict[str, float] = {}
        docs: Dict[str, SearchResult] = {}

        for results in result_lists:
            for rank, result in enumerate(results, start=1):
                doc_id = result.document.id
                rrf_score = 1.0 / (k + rank)
                scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score
                if doc_id not in docs:
                    docs[doc_id] = result

        sorted_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
        output = []
        for new_rank, doc_id in enumerate(sorted_ids, start=1):
            result = docs[doc_id]
            result.score = scores[doc_id]
            result.rank = new_rank
            output.append(result)
        return output

    def retrieve_by_category(
        self,
        query_text: str,
        category: str,
        top_n: Optional[int] = None
    ) -> RetrievalResult:
        """Retrieve documents filtered by specific category.

        Args:
            query_text: User query.
            category: Category to filter by (e.g., "Medication", "Nutrition").
            top_n: Number of results to return.

        Returns:
            RetrievalResult filtered by category.

        Example:
            >>> result = service.retrieve_by_category(
            ...     "ยาแก้ปวด",
            ...     "Medication",
            ...     top_n=5
            ... )
        """
        # Process query
        query = self.query_processor.process(query_text)

        # Override inferred category
        query.inferred_category = category

        # Search with explicit category filter
        candidates = self.hybrid_searcher.search(
            query=query,
            category_filter=category
        )

        # Rerank
        reranked = self.reranker.rerank(query, candidates, top_n=top_n)

        return RetrievalResult(
            query=query,
            candidates=candidates,
            reranked_results=reranked,
            metadata_filters={'category': category}
        )


def retrieve(
    query_text: str,
    settings: Optional[Settings] = None,
    top_k: int = 20,
    top_n: int = 5
) -> RetrievalResult:
    """Convenience function to retrieve documents.

    Creates a RetrievalService and performs retrieval.

    Args:
        query_text: User query.
        settings: Configuration settings. Uses defaults if None.
        top_k: Number of candidates from hybrid search.
        top_n: Number of final results after reranking.

    Returns:
        RetrievalResult with retrieved documents.

    Example:
        >>> result = retrieve("ฉันควรกินยาแก้ปวดไหม")
        >>> for doc in result.reranked_results:
        ...     print(doc.document.content[:50])
    """
    if settings is None:
        from config.settings import get_settings
        settings = get_settings()

    service = RetrievalService(settings)
    return service.retrieve(query_text, top_k=top_k, top_n=top_n)
