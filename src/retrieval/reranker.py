"""Reranking module using BGE-Reranker-v2-m3.

This module implements cross-encoder reranking to refine initial
search results from hybrid search. Provides more accurate relevance
scoring at the cost of increased latency.

BGE-Reranker-v2-m3 is a multilingual cross-encoder that excels at
Thai language relevance scoring.
"""

import logging
from typing import List, Optional

from src.core.types import Query, SearchResult
from src.core.exceptions import RetrievalError


logger = logging.getLogger(__name__)


class BGEReranker:
    """BGE-Reranker-v2-m3 cross-encoder for result reranking.

    Uses a cross-encoder model to compute precise relevance scores
    for (query, document) pairs. More accurate than bi-encoder
    embeddings but slower due to pairwise computation.

    Attributes:
        model_name: HuggingFace model identifier.
        top_n: Number of top results to return after reranking.
        batch_size: Batch size for reranking.
        device: Computation device.
        model: Loaded reranker model.

    Example:
        >>> reranker = BGEReranker()
        >>> reranked = reranker.rerank(query, candidates, top_n=5)
        >>> print(f"Top result: {reranked[0].document.content[:50]}")
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        top_n: int = 5,
        batch_size: int = 32,
        device: Optional[str] = None,
        use_fp16: bool = True
    ):
        """Initialize BGE reranker.

        Args:
            model_name: HuggingFace model identifier.
            top_n: Number of results to return after reranking.
            batch_size: Batch size for processing.
            device: Device ('cuda', 'cpu', or None for auto).
            use_fp16: Use half precision for faster inference.
        """
        self.model_name = model_name
        self.top_n = top_n
        self.batch_size = batch_size
        self.use_fp16 = use_fp16

        # Determine device
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(
            f"Initializing BGE-Reranker: {model_name} on {self.device}"
        )

        # Load model
        self.model = self._load_model()

    def _load_model(self):
        """Load BGE reranker model.

        Returns:
            Loaded FlagReranker instance.

        Raises:
            RetrievalError: If model loading fails.
        """
        try:
            from FlagEmbedding import FlagReranker

            model = FlagReranker(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device
            )

            logger.info(f"Successfully loaded BGE-Reranker on {self.device}")
            return model

        except ImportError:
            error_msg = (
                "FlagEmbedding library required for BGE-Reranker. "
                "Install with: pip install FlagEmbedding"
            )
            logger.error(error_msg)
            raise RetrievalError(error_msg)

        except Exception as e:
            error_msg = f"Failed to load BGE-Reranker: {e}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e

    def rerank(
        self,
        query: Query,
        candidates: List[SearchResult],
        top_n: Optional[int] = None
    ) -> List[SearchResult]:
        """Rerank search results using cross-encoder.

        Args:
            query: Original query object.
            candidates: Initial search results to rerank.
            top_n: Number of top results to return. Uses default if None.

        Returns:
            Reranked list of SearchResult objects (top_n results).

        Raises:
            RetrievalError: If reranking fails.

        Example:
            >>> candidates = hybrid_search(query, db_manager, top_k=20)
            >>> reranked = reranker.rerank(query, candidates, top_n=5)
            >>> for i, result in enumerate(reranked, 1):
            ...     print(f"{i}. Score: {result.score:.4f}")
        """
        if not candidates:
            logger.warning("No candidates provided for reranking")
            return []

        top_n = top_n or self.top_n

        try:
            logger.debug(
                f"Reranking {len(candidates)} candidates to top {top_n}"
            )

            # Prepare (query, document) pairs
            pairs = [
                [query.processed_text, result.document.content]
                for result in candidates
            ]

            # Compute reranking scores
            scores = self.model.compute_score(
                pairs,
                batch_size=self.batch_size,
                normalize=True  # Normalize scores to [0, 1]
            )

            # Handle single result case (returns float instead of list)
            if not isinstance(scores, list):
                scores = [scores]

            # Update SearchResult objects with new scores
            for result, score in zip(candidates, scores):
                result.score = float(score)
                result.score_breakdown['rerank_score'] = float(score)

            # Sort by new scores (descending)
            reranked = sorted(
                candidates,
                key=lambda x: x.score,
                reverse=True
            )

            # Take top_n and update ranks
            top_results = reranked[:top_n]
            for rank, result in enumerate(top_results, start=1):
                result.rank = rank

            logger.info(
                f"Reranked {len(candidates)} to top {len(top_results)} results"
            )

            return top_results

        except Exception as e:
            error_msg = f"Reranking failed: {e}"
            logger.error(error_msg)
            raise RetrievalError(error_msg) from e

    def compute_scores(
        self,
        query_text: str,
        documents: List[str]
    ) -> List[float]:
        """Compute relevance scores for query-document pairs.

        Lower-level method that returns raw scores without
        creating SearchResult objects.

        Args:
            query_text: Query text.
            documents: List of document texts.

        Returns:
            List of relevance scores (0.0 to 1.0).

        Example:
            >>> scores = reranker.compute_scores(
            ...     "ฉันควรกินยา",
            ...     ["เอกสาร 1", "เอกสาร 2"]
            ... )
        """
        try:
            pairs = [[query_text, doc] for doc in documents]

            scores = self.model.compute_score(
                pairs,
                batch_size=self.batch_size,
                normalize=True
            )

            # Handle single result
            if not isinstance(scores, list):
                scores = [scores]

            return [float(s) for s in scores]

        except Exception as e:
            logger.error(f"Score computation failed: {e}")
            return [0.0] * len(documents)


def rerank_results(
    query: Query,
    candidates: List[SearchResult],
    model_name: str = "BAAI/bge-reranker-v2-m3",
    top_n: int = 5
) -> List[SearchResult]:
    """Convenience function to rerank search results.

    Args:
        query: Query object.
        candidates: Search results to rerank.
        model_name: Reranker model name.
        top_n: Number of top results to return.

    Returns:
        Reranked list of SearchResult objects.

    Example:
        >>> reranked = rerank_results(query, candidates, top_n=5)
    """
    reranker = BGEReranker(model_name=model_name, top_n=top_n)
    return reranker.rerank(query, candidates, top_n=top_n)
