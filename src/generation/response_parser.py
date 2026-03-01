"""Response parser for LLM output processing.

This module parses LLM responses and generates citations from
retrieval results for medical accuracy and traceability.
"""

import logging
from typing import List

from src.core.types import RetrievalResult, GeneratedResponse


logger = logging.getLogger(__name__)


class ResponseParser:
    """Parser for processing LLM responses with retrieval-based citations.

    Citations are generated directly from reranked retrieval results
    (source_file + page_number), not extracted from LLM output.

    Example:
        >>> parser = ResponseParser()
        >>> parsed = parser.parse_response(
        ...     llm_response="คำตอบจาก LLM",
        ...     retrieval_result=result,
        ...     query_text="คำถาม"
        ... )
        >>> print(parsed.citations)
    """

    def __init__(
        self,
        strict_validation: bool = False,
        require_citations: bool = True
    ):
        self.strict_validation = strict_validation
        self.require_citations = require_citations
        logger.info(
            f"ResponseParser initialized: "
            f"strict={strict_validation}, require_citations={require_citations}"
        )

    def parse_response(
        self,
        llm_response: str,
        retrieval_result: RetrievalResult,
        query_text: str,
        generation_time_ms: float = 0.0,
        model_name: str = "deepseek-chat"
    ) -> GeneratedResponse:
        """Parse LLM response and create GeneratedResponse object.

        Citations are generated from retrieval_result.reranked_results,
        not extracted from the LLM response text.

        Args:
            llm_response: Raw text from LLM.
            retrieval_result: Original retrieval result with context.
            query_text: Original user query.
            generation_time_ms: Time taken for generation.
            model_name: Name of model used.

        Returns:
            GeneratedResponse with answer and retrieval-based citations.
        """
        context_docs = [
            result.document for result in retrieval_result.reranked_results
        ]

        # Generate citations from retrieval results (deduplicated)
        seen = set()
        citations: List[str] = []
        for result in retrieval_result.reranked_results:
            doc = result.document
            citation = f"Source: {doc.source_file}, Page: {doc.page_number}"
            if citation not in seen:
                seen.add(citation)
                citations.append(citation)

        response = GeneratedResponse(
            query_text=query_text,
            answer=llm_response,
            citations=citations,
            context_used=context_docs,
            generation_time_ms=generation_time_ms,
            model_name=model_name
        )

        logger.info(
            f"Parsed response: {len(citations)} citations, "
            f"{len(llm_response)} chars, {generation_time_ms:.2f}ms"
        )

        return response


def create_response_parser(
    strict_validation: bool = False,
    require_citations: bool = True
) -> ResponseParser:
    """Factory function to create ResponseParser."""
    return ResponseParser(
        strict_validation=strict_validation,
        require_citations=require_citations
    )
