"""Generation engine for RAG-based medical Q&A.

This module orchestrates the complete generation pipeline:
1. Receive retrieval results
2. Build prompt with context
3. Sanitize query (PII removal)
4. Call LLM
5. Parse and validate response

CRITICAL: System NEVER invents medical advice. If context is insufficient,
it refuses to answer.
"""

import logging
import re
import time
from typing import Optional, Dict, Any

from src.core.types import RetrievalResult, GeneratedResponse
from src.core.exceptions import GenerationError
from src.generation.client import DeepSeekClient
from src.generation.prompt_builder import PromptBuilder
from src.generation.response_parser import ResponseParser


logger = logging.getLogger(__name__)


class GenerationEngine:
    """Main engine for medical response generation.

    Orchestrates the complete generation pipeline with medical safety constraints:
    - Only answers based on provided context
    - Enforces citation requirements
    - Strips PII before API calls
    - Handles insufficient context gracefully

    Attributes:
        llm_client: DeepSeek API client.
        prompt_builder: Prompt construction engine.
        response_parser: Citation extraction and validation.
        min_context_score: Minimum score threshold for generation.
        min_context_count: Minimum number of context docs required.

    Example:
        >>> engine = GenerationEngine(client, builder, parser)
        >>> response = engine.generate(retrieval_result, "ฉันควรกินยาไหม")
        >>> print(response.answer)
    """

    # PII patterns for basic sanitization
    # Thai ID: 13 digits
    THAI_ID_PATTERN = re.compile(r'\b\d{1}-?\d{4}-?\d{5}-?\d{2}-?\d{1}\b')
    # Phone: 10 digits starting with 0
    PHONE_PATTERN = re.compile(r'\b0\d{1,2}-?\d{3,4}-?\d{4}\b')
    # Email
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

    def __init__(
        self,
        llm_client: DeepSeekClient,
        prompt_builder: PromptBuilder,
        response_parser: ResponseParser,
        min_context_score: float = 0.5,
        min_context_count: int = 1,
        enable_pii_removal: bool = True,
        fallback_message: str = "ขออภัยครับ/ค่ะ ไม่มีข้อมูลเพียงพอในระบบ กรุณาปรึกษาทันตแพทย์โดยตรงค่ะ"
    ):
        """Initialize generation engine.

        Args:
            llm_client: DeepSeek client for API calls.
            prompt_builder: Prompt builder for formatting.
            response_parser: Response parser for citation validation.
            min_context_score: Minimum relevance score for context.
            min_context_count: Minimum number of context documents.
            enable_pii_removal: Whether to strip PII from queries.
            fallback_message: Thai message when context is insufficient.
        """
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.response_parser = response_parser
        self.min_context_score = min_context_score
        self.min_context_count = min_context_count
        self.enable_pii_removal = enable_pii_removal
        self.fallback_message = fallback_message

        logger.info(
            f"GenerationEngine initialized: "
            f"min_score={min_context_score}, min_count={min_context_count}"
        )

    def generate(
        self,
        retrieval_result: RetrievalResult,
        user_query: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        patient_context: Optional[Dict[str, Any]] = None
    ) -> GeneratedResponse:
        """Generate response based on retrieval results.

        Complete pipeline:
        1. Validate context quality
        2. Sanitize query (remove PII)
        3. Build prompt with context
        4. Call LLM
        5. Parse and validate response

        Args:
            retrieval_result: Results from retrieval pipeline.
            user_query: Original query. Uses query from retrieval_result if None.
            temperature: Override LLM temperature.
            max_tokens: Override max tokens.
            patient_context: Optional patient data (not used in Phase 1, for future use).

        Returns:
            GeneratedResponse with answer and citations.

        Raises:
            GenerationError: If generation fails.

        Example:
            >>> response = engine.generate(
            ...     retrieval_result,
            ...     "ฉันควรกินยาแก้ปวดไหม"
            ... )
            >>> print(response.answer)
            >>> print(f"Citations: {response.citations}")

        Note:
            In Phase 1, patient context is already embedded in retrieval_result.query,
            so the patient_context parameter is not used. Added for API consistency.
        """
        start_time = time.time()

        # Get query text
        if user_query is None:
            user_query = retrieval_result.query.original_text

        try:
            # Step 1: Validate context quality
            logger.info("Validating context quality...")
            if not self._validate_context(retrieval_result):
                logger.warning("Insufficient context, returning fallback")
                return self._create_fallback_response(
                    user_query,
                    retrieval_result,
                    time.time() - start_time
                )

            # Step 2: Sanitize query (remove PII)
            sanitized_query = user_query
            if self.enable_pii_removal:
                sanitized_query = self._remove_pii(user_query)
                if sanitized_query != user_query:
                    logger.info("PII removed from query")

            # Step 3: Build prompt
            logger.info("Building prompt with context...")
            messages = self.prompt_builder.build_prompt(
                retrieval_result,
                sanitized_query
            )

            # Step 4: Call LLM
            logger.info("Calling LLM API...")
            llm_response = self.llm_client.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Validate LLM response
            if not self.llm_client.validate_response(llm_response):
                logger.error("Invalid LLM response")
                return self._create_fallback_response(
                    user_query,
                    retrieval_result,
                    time.time() - start_time
                )

            # Step 5: Parse and validate response
            logger.info("Parsing response and validating citations...")
            parsed_response = self.response_parser.parse_response(
                llm_response=llm_response["content"],
                retrieval_result=retrieval_result,
                query_text=user_query,
                generation_time_ms=(time.time() - start_time) * 1000,
                model_name=llm_response.get("model", "deepseek-chat")
            )

            # Check citation quality
            quality = self.response_parser.check_citation_quality(
                llm_response["content"],
                retrieval_result
            )

            if not quality["has_citations"]:
                logger.warning("Response has no citations (medical safety concern)")

            if quality["invalid_citations"]:
                logger.warning(
                    f"Response has invalid citations: {quality['invalid_citations']}"
                )

            logger.info(
                f"Generation complete: {len(parsed_response.answer)} chars, "
                f"{len(parsed_response.citations)} citations, "
                f"{parsed_response.generation_time_ms:.2f}ms"
            )

            return parsed_response

        except Exception as e:
            error_msg = f"Generation failed: {e}"
            logger.error(error_msg)

            # Return fallback instead of raising error
            logger.info("Returning fallback response due to error")
            return self._create_fallback_response(
                user_query,
                retrieval_result,
                time.time() - start_time,
                error=str(e)
            )

    def generate_with_q2d(
        self,
        query: str,
        q2d_temperature: float = 0.3
    ) -> str:
        """Generate expanded query using Q2D (Query to Document).

        Args:
            query: User query.
            q2d_temperature: Temperature for query expansion.

        Returns:
            Expanded query with related terms in Thai.

        Example:
            >>> expanded = engine.generate_with_q2d("ฉันควรกินอาหารอะไร")
            >>> # Use expanded query for retrieval
        """
        logger.info(f"Generating Q2D for query: '{query[:50]}...'")

        try:
            # Build Q2D prompt
            messages = self.prompt_builder.build_q2d_prompt(query)

            # Generate expanded query
            response = self.llm_client.generate(
                messages=messages,
                temperature=q2d_temperature,
                max_tokens=6000
            )

            q2d_text = response["content"].strip()
            logger.info(f"Q2D generated: {len(q2d_text)} chars")

            return q2d_text

        except Exception as e:
            logger.error(f"Q2D generation failed: {e}")
            # Return original query as fallback
            return query

    def _validate_context(self, retrieval_result: RetrievalResult) -> bool:
        """Validate that context is sufficient for generation.

        Args:
            retrieval_result: Retrieval result to validate.

        Returns:
            True if context quality is acceptable.
        """
        results = retrieval_result.reranked_results

        # Check if we have any results
        if not results:
            logger.warning("No context available")
            return False

        # Check if we have minimum number of results
        if len(results) < self.min_context_count:
            logger.warning(
                f"Only {len(results)} results, need at least {self.min_context_count}"
            )
            return False

        # Check if top result meets minimum score
        top_score = results[0].score
        if top_score < self.min_context_score:
            logger.warning(
                f"Top result score {top_score:.2f} below threshold "
                f"{self.min_context_score}"
            )
            return False

        return True

    def _remove_pii(self, text: str) -> str:
        """Remove PII (Personally Identifiable Information) from text.

        Masks:
        - Thai ID numbers (13 digits)
        - Phone numbers (10 digits starting with 0)
        - Email addresses

        Args:
            text: Text potentially containing PII.

        Returns:
            Text with PII masked.

        Example:
            >>> sanitized = engine._remove_pii("โทร 081-234-5678")
            >>> print(sanitized)  # "โทร [PHONE]"
        """
        # Mask Thai ID
        text = self.THAI_ID_PATTERN.sub('[THAI_ID]', text)

        # Mask phone numbers
        text = self.PHONE_PATTERN.sub('[PHONE]', text)

        # Mask emails
        text = self.EMAIL_PATTERN.sub('[EMAIL]', text)

        return text

    def _create_fallback_response(
        self,
        query_text: str,
        retrieval_result: RetrievalResult,
        elapsed_time: float,
        error: Optional[str] = None
    ) -> GeneratedResponse:
        """Create fallback response when generation fails or context is insufficient.

        Args:
            query_text: Original user query.
            retrieval_result: Retrieval result (may be empty).
            elapsed_time: Time elapsed in seconds.
            error: Optional error message.

        Returns:
            GeneratedResponse with fallback message.
        """
        # Use context from retrieval if available
        context_docs = []
        if retrieval_result.reranked_results:
            context_docs = [r.document for r in retrieval_result.reranked_results]

        response = GeneratedResponse(
            query_text=query_text,
            answer=self.fallback_message,
            citations=[],
            context_used=context_docs,
            generation_time_ms=elapsed_time * 1000,
            model_name=self.llm_client.model_name
        )

        if error:
            logger.error(f"Fallback response due to error: {error}")

        return response

    def batch_generate(
        self,
        retrieval_results: list,
        user_queries: Optional[list] = None
    ) -> list:
        """Generate responses for multiple queries.

        Args:
            retrieval_results: List of RetrievalResult objects.
            user_queries: List of query strings. Uses queries from results if None.

        Returns:
            List of GeneratedResponse objects.

        Example:
            >>> responses = engine.batch_generate(
            ...     [result1, result2, result3]
            ... )
            >>> for resp in responses:
            ...     print(resp.answer)
        """
        if user_queries is None:
            user_queries = [None] * len(retrieval_results)

        responses = []

        for i, (result, query) in enumerate(zip(retrieval_results, user_queries), 1):
            logger.info(f"Generating response {i}/{len(retrieval_results)}")

            response = self.generate(result, query)
            responses.append(response)

        logger.info(f"Batch generation complete: {len(responses)} responses")

        return responses

    def get_config(self) -> Dict[str, Any]:
        """Get current engine configuration.

        Returns:
            Dict with configuration details.

        Example:
            >>> config = engine.get_config()
            >>> print(f"Min score: {config['min_context_score']}")
        """
        return {
            "llm_model": self.llm_client.model_name,
            "temperature": self.llm_client.temperature,
            "max_tokens": self.llm_client.max_tokens,
            "min_context_score": self.min_context_score,
            "min_context_count": self.min_context_count,
            "enable_pii_removal": self.enable_pii_removal,
            "fallback_message": self.fallback_message
        }


def create_generation_engine(
    llm_config: Dict[str, Any],
    prompts_config: Dict[str, Any],
    api_key: str,
    min_context_score: float = 0.5,
    min_context_count: int = 1
) -> GenerationEngine:
    """Factory function to create GenerationEngine from configuration.

    Args:
        llm_config: LLM configuration from model_config.yaml.
        prompts_config: Prompts configuration from prompts.yaml.
        api_key: DeepSeek API key.
        min_context_score: Minimum context relevance score.
        min_context_count: Minimum number of context documents.

    Returns:
        Initialized GenerationEngine.

    Example:
        >>> from config.settings import get_settings
        >>> settings = get_settings()
        >>> engine = create_generation_engine(
        ...     settings.model_config['llm'],
        ...     settings.prompts_config,
        ...     api_key="sk-xxx"
        ... )
    """
    from src.generation.client import create_client_from_config
    from src.generation.prompt_builder import create_prompt_builder
    from src.generation.response_parser import create_response_parser

    # Create components
    client = create_client_from_config(llm_config, api_key)
    builder = create_prompt_builder(prompts_config)
    parser = create_response_parser(
        strict_validation=False,  # Don't raise errors for invalid citations
        require_citations=True    # Warn if no citations
    )

    # Create engine
    engine = GenerationEngine(
        llm_client=client,
        prompt_builder=builder,
        response_parser=parser,
        min_context_score=min_context_score,
        min_context_count=min_context_count
    )

    return engine
