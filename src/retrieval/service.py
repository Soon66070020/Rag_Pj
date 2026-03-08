"""Retrieval service facade.

This module provides a high-level interface to the retrieval pipeline,
combining query processing, hybrid search, and reranking into a single
function call.

Optimized for low latency and high relevance (Recall@K).
"""

import logging
import time
from typing import Optional, Dict, Any, Tuple

from src.core.types import Query, RetrievalResult
from src.core.exceptions import RetrievalError
from src.retrieval.query_processor import QueryProcessor
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import BGEReranker
from src.retrieval.query_guard import QueryGuard, QueryGuardResult
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
            device=reranker_config.get('device', None),
            use_fp16=reranker_config.get('use_fp16', True)
        )

        # Initialize Query Guard (multi-LLM pipeline)
        guard_config = settings.model_config.get('query_guard', {})
        if guard_config:
            logger.info("Initializing Query Guard (deepseek-chat)...")
            guard_client = DeepSeekClient(
                api_key=settings.deepseek_api_key,
                model_name=guard_config.get('model_name', 'deepseek-chat'),
                api_base=guard_config.get('api_base', 'https://api.deepseek.com'),
                temperature=guard_config.get('temperature', 0.1),
                max_tokens=guard_config.get('max_tokens', 4096),
                timeout=guard_config.get('timeout', 30),
            )
            self.query_guard: Optional[QueryGuard] = QueryGuard(
                llm_client=guard_client,
                prompts=settings.prompts,
            )
            logger.info("Query Guard ready")
        else:
            self.query_guard = None
            logger.info("Query Guard not configured, skipping")

        logger.info(
            f"RetrievalService ready: alpha={self.alpha}, "
            f"top_k={self.top_k}, top_n={self.top_n}"
        )

    def retrieve(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        use_reranking: bool = True,
        patient_context: Optional[Dict[str, Any]] = None
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
            patient_context: Optional patient data (summary, flows) for Phase 1.

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
            # Step 1: Process query with patient context
            logger.info(f"Processing query: '{query_text[:50]}...'")
            query = self.query_processor.process(query_text, patient_context=patient_context)

            # Determine category filter
            category_filter = None
            if self.use_category_filter and query.inferred_category:
                category_filter = query.inferred_category
                logger.debug(f"Using category filter: {category_filter}")

            # Step 2: Hybrid search
            logger.info(f"Executing hybrid search (top_k={top_k or self.top_k})...")
            candidates = self.hybrid_searcher.search(
                query=query,
                top_k=top_k,
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

    def retrieve_with_guard(
        self,
        query_text: str,
        patient_context: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
    ) -> Tuple[QueryGuardResult, Optional[RetrievalResult]]:
        """Retrieve with multi-LLM query guard pipeline.

        Pipeline:
        1. QueryGuard: scope check → clarification/rewrite → classification
        2. If out-of-domain: return early (no retrieval)
        3. Multi-category search using classification results
        4. Rerank with BGE cross-encoder

        Args:
            query_text: User query in Thai.
            patient_context: Full patient data dict (patient, flows, summary).
            top_k: Number of candidates from hybrid search.
            top_n: Number of final results after reranking.

        Returns:
            Tuple of (QueryGuardResult, Optional[RetrievalResult]).
            RetrievalResult is None when query is out-of-domain.
        """
        if self.query_guard is None:
            # No guard configured — fall back to normal retrieve
            logger.warning("QueryGuard not configured, using standard retrieve")
            guard_result = QueryGuardResult(
                is_in_scope=True,
                clarification_needed=False,
                original_query=query_text,
                effective_query=query_text,
            )
            retrieval_result = self.retrieve(
                query_text,
                top_k=top_k,
                top_n=top_n,
                patient_context=patient_context,
            )
            return guard_result, retrieval_result

        # Run guard pipeline
        guard_result = self.query_guard.run(query_text, patient_context)

        # Out of domain → return early, no retrieval
        if not guard_result.is_in_scope:
            logger.info("Query out of domain, skipping retrieval")
            return guard_result, None

        # Use effective_query (may be rewritten by LLM2)
        effective_query = guard_result.effective_query
        start_time = time.time()

        try:
            # Process query with embeddings
            query = self.query_processor.process(
                effective_query, patient_context=patient_context
            )

            # Determine search strategy based on classification
            categories = []
            if guard_result.classification:
                categories = [
                    c.category for c in guard_result.classification.categories
                ]

            # Multi-category search with fallback
            if categories:
                logger.info(f"Multi-category search: {categories}")
                candidates = self.hybrid_searcher.search_multi_category(
                    query=query,
                    categories=categories,
                    top_k=top_k,
                )
            else:
                # No classification → standard search
                candidates = self.hybrid_searcher.search(
                    query=query, top_k=top_k
                )

            if not candidates:
                logger.warning("No candidates found")
                return guard_result, RetrievalResult(
                    query=query,
                    candidates=[],
                    reranked_results=[],
                    metadata_filters={'categories': categories},
                    retrieval_time_ms=(time.time() - start_time) * 1000,
                )

            # Rerank
            reranked = self.reranker.rerank(query, candidates, top_n=top_n)

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Guard retrieval complete: {len(reranked)} results "
                f"in {elapsed_ms:.0f}ms"
            )

            retrieval_result = RetrievalResult(
                query=query,
                candidates=candidates,
                reranked_results=reranked,
                metadata_filters={'categories': categories},
                retrieval_time_ms=elapsed_ms,
            )
            return guard_result, retrieval_result

        except Exception as e:
            error_msg = f"Retrieval with guard failed: {e}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e


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
