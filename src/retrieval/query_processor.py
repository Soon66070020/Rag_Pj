"""Query processing module for the retrieval pipeline.

This module handles query preprocessing including:
- Thai text normalization
- Q2D (Query to Document) query expansion with related terms
- Query embedding generation (BGE-M3)
- Category inference from Thai keywords

Optimized for low latency while maintaining high quality embeddings.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

from src.core.types import Query
from src.core.exceptions import QueryProcessingError
from src.utils.text_processing import (
    normalize_thai_text,
    infer_category_from_thai_keywords,
    clean_thai_text_for_embedding
)
from src.knowledge_base.embedding_generator import BGEEmbeddingGenerator


logger = logging.getLogger(__name__)


class QueryProcessor:
    """Process user queries for retrieval pipeline.

    Handles Thai text normalization, Q2D expansion, embedding generation,
    and category inference to prepare queries for hybrid search.

    Attributes:
        embedding_generator: BGE-M3 embedding generator.
        auto_infer_category: Whether to automatically infer category.
        q2d_client: Optional DeepSeek client for Q2D generation.
        prompt_builder: Optional prompt builder for Q2D prompts.

    Example:
        >>> processor = QueryProcessor()
        >>> query = processor.process("ฉันควรกินยาแก้ปวดไหม")
        >>> print(query.inferred_category)
        'Medication'
    """

    def __init__(
        self,
        embedding_generator: Optional[BGEEmbeddingGenerator] = None,
        auto_infer_category: bool = True,
        q2d_client=None,
        prompt_builder=None
    ):
        """Initialize query processor.

        Args:
            embedding_generator: BGE-M3 generator. Creates new if None.
            auto_infer_category: Whether to infer category from keywords.
            q2d_client: DeepSeek client for Q2D generation. None to disable Q2D.
            prompt_builder: PromptBuilder for building Q2D prompts.
        """
        self.auto_infer_category = auto_infer_category
        self.q2d_client = q2d_client
        self.prompt_builder = prompt_builder

        # Initialize embedding generator
        if embedding_generator is None:
            logger.info("Initializing BGE-M3 embedding generator for queries")
            self.embedding_generator = BGEEmbeddingGenerator(
                batch_size=1  # Single query at a time for low latency
            )
        else:
            self.embedding_generator = embedding_generator

        logger.info(f"QueryProcessor initialized (Q2D: {'enabled' if q2d_client else 'disabled'})")

    def process(self, query_text: str) -> Query:
        """Process a user query.

        Complete processing pipeline:
        1. Normalize Thai text
        2. Clean for embedding
        3. Generate embeddings (dense + sparse)
        4. Infer category from Thai keywords

        Args:
            query_text: Raw user query in Thai or mixed Thai-English.

        Returns:
            Query object with all fields populated.

        Raises:
            QueryProcessingError: If processing fails.

        Example:
            >>> query = processor.process("ฉันควรกินอะไรหลังผ่าตัด")
            >>> print(f"Category: {query.inferred_category}")
            'Nutrition'
        """
        if not query_text or not query_text.strip():
            raise QueryProcessingError("Empty query provided")

        try:
            # Step 1: Normalize Thai text
            processed_text = self._normalize_query(query_text)

            # Step 2: Infer category (fast, keyword-based)
            category = None
            if self.auto_infer_category:
                category = self._infer_category(processed_text)

            # Step 3: Q2D - Expand query with related terms
            q2d_expansion = ""
            embedding_text = processed_text  # Default: embed original query
            if self.q2d_client and self.prompt_builder:
                q2d_expansion = self._generate_q2d(processed_text)
                if q2d_expansion and q2d_expansion != processed_text:
                    embedding_text = q2d_expansion  # Embed expanded query instead
                    logger.info(f"Using Q2D expansion for embedding ({len(q2d_expansion)} chars)")

            # Step 4: Generate embeddings (from Q2D expansion or original query)
            dense_vector, sparse_vector = self._generate_embeddings(embedding_text)

            # Create Query object
            query = Query(
                original_text=query_text,
                processed_text=processed_text,
                q2d_expansion=q2d_expansion,
                inferred_category=category,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                timestamp=datetime.now()
            )

            logger.debug(
                f"Processed query: '{query_text[:50]}...' "
                f"(category: {category}, q2d: {'yes' if q2d_expansion else 'no'})"
            )

            return query

        except Exception as e:
            error_msg = f"Failed to process query: {e}"
            logger.error(error_msg)
            raise QueryProcessingError(error_msg) from e

    def _normalize_query(self, text: str) -> str:
        """Normalize Thai query text.

        Args:
            text: Raw query text.

        Returns:
            Normalized text.
        """
        # Basic normalization
        text = normalize_thai_text(text)

        # Additional cleaning for embedding
        text = clean_thai_text_for_embedding(text)

        return text

    def _infer_category(self, text: str) -> Optional[str]:
        """Infer query category from Thai keywords.

        Args:
            text: Normalized query text.

        Returns:
            Inferred category or None.
        """
        try:
            category = infer_category_from_thai_keywords(text)
            logger.debug(f"Inferred category: {category}")
            return category
        except Exception as e:
            logger.warning(f"Category inference failed: {e}")
            return None

    def _generate_q2d(self, query_text: str) -> str:
        """Generate expanded query using Q2D (Query to Document).

        Args:
            query_text: Normalized query text.

        Returns:
            Expanded query with related terms, or original query on failure.
        """
        try:
            logger.info(f"Generating Q2D for: '{query_text[:50]}...'")
            messages = self.prompt_builder.build_q2d_prompt(query_text)
            response = self.q2d_client.generate(messages=messages)
            q2d_text = response["content"].strip()
            logger.info(f"Q2D generated: {len(q2d_text)} chars")
            return q2d_text
        except Exception as e:
            logger.warning(f"Q2D generation failed, using original query: {e}")
            return query_text

    def _generate_embeddings(self, text: str) -> tuple:
        """Generate query embeddings.

        Args:
            text: Cleaned query text.

        Returns:
            Tuple of (dense_vector, sparse_vector).
        """
        try:
            # Generate embeddings (single query for low latency)
            embedding = self.embedding_generator.encode_single(
                text,
                return_dense=True,
                return_sparse=True,
                clean_text=False  # Already cleaned
            )

            dense_vector = embedding.get('dense', [])
            sparse_vector = embedding.get('sparse', {})

            # Convert numpy array to list if needed
            if hasattr(dense_vector, 'tolist'):
                dense_vector = dense_vector.tolist()

            return dense_vector, sparse_vector

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise QueryProcessingError(f"Failed to generate embeddings: {e}") from e

    def process_batch(self, queries: list[str]) -> list[Query]:
        """Process multiple queries in batch for efficiency.

        Args:
            queries: List of query strings.

        Returns:
            List of Query objects.

        Example:
            >>> queries = ["คำถาม 1", "คำถาม 2"]
            >>> results = processor.process_batch(queries)
        """
        processed_queries = []

        for query_text in queries:
            try:
                query = self.process(query_text)
                processed_queries.append(query)
            except QueryProcessingError as e:
                logger.error(f"Failed to process query '{query_text}': {e}")
                # Continue processing remaining queries

        return processed_queries


def process_query(
    query_text: str,
    embedding_generator: Optional[BGEEmbeddingGenerator] = None
) -> Query:
    """Convenience function to process a query.

    Args:
        query_text: Raw query text.
        embedding_generator: Optional embedding generator.

    Returns:
        Processed Query object.

    Example:
        >>> query = process_query("ฉันควรกินยาแก้ปวดไหม")
        >>> print(query.inferred_category)
    """
    processor = QueryProcessor(embedding_generator)
    return processor.process(query_text)
