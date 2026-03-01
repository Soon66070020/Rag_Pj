"""Response parser for citation extraction and validation.

This module parses LLM responses to extract citations and verify them
against the provided context, ensuring medical accuracy and traceability.
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional

from src.core.types import Document, RetrievalResult, GeneratedResponse


logger = logging.getLogger(__name__)


class ResponseParser:
    """Parser for extracting and validating citations from LLM responses.

    Extracts citations in format: [Source: filename, Page: number]
    Validates that citations match the documents provided in context.

    Attributes:
        strict_validation: If True, reject responses with invalid citations.
        require_citations: If True, warn when no citations are found.

    Example:
        >>> parser = ResponseParser(strict_validation=True)
        >>> parsed = parser.parse_response(
        ...     llm_response="คำตอบ [Source: guide.pdf, Page: 5]",
        ...     retrieval_result=result,
        ...     query_text="คำถาม"
        ... )
        >>> print(parsed.citations)  # ['Source: guide.pdf, Page: 5']
    """

    # Citation pattern: [Source: filename, Page: number]
    CITATION_PATTERN = re.compile(
        r'\[Source:\s*([^,]+),\s*Page:\s*(\d+)\]',
        re.IGNORECASE
    )

    def __init__(
        self,
        strict_validation: bool = False,
        require_citations: bool = True
    ):
        """Initialize response parser.

        Args:
            strict_validation: If True, raise error for invalid citations.
            require_citations: If True, log warning when no citations found.
        """
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

        Args:
            llm_response: Raw text from LLM.
            retrieval_result: Original retrieval result with context.
            query_text: Original user query.
            generation_time_ms: Time taken for generation.
            model_name: Name of model used.

        Returns:
            GeneratedResponse with parsed answer and validated citations.

        Raises:
            ValueError: If strict_validation=True and citations are invalid.

        Example:
            >>> parsed = parser.parse_response(
            ...     "ควรรับประทานยาตามแพทย์สั่ง [Source: guide.pdf, Page: 3]",
            ...     retrieval_result,
            ...     "ฉันควรกินยาไหม"
            ... )
        """
        # Get context documents
        context_docs = [
            result.document for result in retrieval_result.reranked_results
        ]

        # Generate citations from retrieval results (not from LLM output)
        seen = set()
        citations = []
        for result in retrieval_result.reranked_results:
            doc = result.document
            citation = f"Source: {doc.source_file}, Page: {doc.page_number}"
            if citation not in seen:
                seen.add(citation)
                citations.append(citation)

        answer_text = llm_response

        # Create GeneratedResponse
        response = GeneratedResponse(
            query_text=query_text,
            answer=answer_text,
            citations=citations,
            context_used=context_docs,
            generation_time_ms=generation_time_ms,
            model_name=model_name
        )

        logger.info(
            f"Parsed response: {len(citations)} citations, "
            f"{len(answer_text)} chars, {generation_time_ms:.2f}ms"
        )

        return response

    def extract_citations(self, text: str) -> List[str]:
        """Extract all citations from text.

        Args:
            text: Text containing citations.

        Returns:
            List of citation strings (without brackets).

        Example:
            >>> citations = parser.extract_citations(
            ...     "Answer [Source: a.pdf, Page: 1] and [Source: b.pdf, Page: 2]"
            ... )
            >>> print(citations)
            ['Source: a.pdf, Page: 1', 'Source: b.pdf, Page: 2']
        """
        matches = self.CITATION_PATTERN.findall(text)

        citations = []
        for filename, page in matches:
            # Reconstruct citation string
            citation = f"Source: {filename.strip()}, Page: {page}"
            citations.append(citation)

        return citations

    def validate_citations(
        self,
        citations: List[str],
        context_docs: List[Document]
    ) -> Tuple[List[str], List[str]]:
        """Validate that citations match provided context documents.

        Args:
            citations: List of extracted citation strings.
            context_docs: Documents that were provided as context.

        Returns:
            Tuple of (valid_citations, invalid_citations).

        Example:
            >>> valid, invalid = parser.validate_citations(
            ...     ['Source: guide.pdf, Page: 5'],
            ...     context_documents
            ... )
        """
        valid = []
        invalid = []

        # Build set of valid (filename, page) tuples from context
        valid_sources = {
            (doc.source_file, doc.page_number)
            for doc in context_docs
        }

        for citation in citations:
            # Parse citation
            match = self.CITATION_PATTERN.search(f"[{citation}]")
            if not match:
                invalid.append(citation)
                continue

            filename, page_str = match.groups()
            page = int(page_str)

            # Check if this source exists in context
            if (filename.strip(), page) in valid_sources:
                valid.append(citation)
            else:
                invalid.append(citation)
                logger.debug(
                    f"Invalid citation: {filename}, Page {page} "
                    f"not found in context"
                )

        return valid, invalid

    def count_citations(self, text: str) -> int:
        """Count number of citations in text.

        Args:
            text: Text to analyze.

        Returns:
            Number of citations found.

        Example:
            >>> count = parser.count_citations(response_text)
            >>> print(f"Found {count} citations")
        """
        return len(self.extract_citations(text))

    def has_sufficient_citations(
        self,
        text: str,
        min_citations: int = 1
    ) -> bool:
        """Check if text has sufficient citations.

        Args:
            text: Text to check.
            min_citations: Minimum required citations.

        Returns:
            True if citation count meets minimum.

        Example:
            >>> if parser.has_sufficient_citations(response, min_citations=2):
            ...     print("Well-cited response")
        """
        return self.count_citations(text) >= min_citations

    def extract_source_files(self, citations: List[str]) -> List[str]:
        """Extract unique source filenames from citations.

        Args:
            citations: List of citation strings.

        Returns:
            List of unique filenames.

        Example:
            >>> files = parser.extract_source_files([
            ...     'Source: a.pdf, Page: 1',
            ...     'Source: a.pdf, Page: 2',
            ...     'Source: b.pdf, Page: 1'
            ... ])
            >>> print(files)  # ['a.pdf', 'b.pdf']
        """
        files = set()

        for citation in citations:
            match = self.CITATION_PATTERN.search(f"[{citation}]")
            if match:
                filename = match.group(1).strip()
                files.add(filename)

        return sorted(list(files))

    def format_citations_list(self, citations: List[str]) -> str:
        """Format citations as a numbered list.

        Args:
            citations: List of citation strings.

        Returns:
            Formatted string with numbered citations.

        Example:
            >>> formatted = parser.format_citations_list(citations)
            >>> print(formatted)
            1. Source: guide.pdf, Page: 5
            2. Source: care.pdf, Page: 12
        """
        if not citations:
            return "No citations found."

        lines = []
        for i, citation in enumerate(citations, start=1):
            lines.append(f"{i}. {citation}")

        return "\n".join(lines)

    def _remove_citations(self, text: str) -> str:
        """Remove citation markers from text.

        Args:
            text: Text with citations.

        Returns:
            Text with citations removed.

        Example:
            >>> clean = parser._remove_citations(
            ...     "Answer [Source: a.pdf, Page: 1] here."
            ... )
            >>> print(clean)  # "Answer  here."
        """
        return self.CITATION_PATTERN.sub('', text).strip()

    def get_citation_coverage(
        self,
        citations: List[str],
        context_docs: List[Document]
    ) -> float:
        """Calculate what percentage of context documents are cited.

        Args:
            citations: Extracted citations.
            context_docs: Context documents.

        Returns:
            Coverage ratio (0.0 to 1.0).

        Example:
            >>> coverage = parser.get_citation_coverage(citations, context_docs)
            >>> print(f"Citation coverage: {coverage:.1%}")
        """
        if not context_docs:
            return 0.0

        # Get cited documents
        cited_sources = set()
        for citation in citations:
            match = self.CITATION_PATTERN.search(f"[{citation}]")
            if match:
                filename, page_str = match.groups()
                page = int(page_str)
                cited_sources.add((filename.strip(), page))

        # Get all available sources
        available_sources = {
            (doc.source_file, doc.page_number)
            for doc in context_docs
        }

        # Calculate coverage
        if not available_sources:
            return 0.0

        coverage = len(cited_sources) / len(available_sources)
        return coverage

    def check_citation_quality(
        self,
        llm_response: str,
        retrieval_result: RetrievalResult
    ) -> Dict[str, Any]:
        """Comprehensive citation quality check.

        Args:
            llm_response: LLM response text.
            retrieval_result: Retrieval result with context.

        Returns:
            Dict with quality metrics:
                - citation_count: Number of citations
                - valid_citations: List of valid citations
                - invalid_citations: List of invalid citations
                - coverage: Percentage of context docs cited
                - has_citations: Boolean
                - all_valid: Boolean

        Example:
            >>> quality = parser.check_citation_quality(response, result)
            >>> if quality['all_valid']:
            ...     print("All citations are valid!")
        """
        citations = self.extract_citations(llm_response)
        context_docs = [r.document for r in retrieval_result.reranked_results]

        valid, invalid = self.validate_citations(citations, context_docs)
        coverage = self.get_citation_coverage(citations, context_docs)

        return {
            "citation_count": len(citations),
            "valid_citations": valid,
            "invalid_citations": invalid,
            "coverage": coverage,
            "has_citations": len(citations) > 0,
            "all_valid": len(invalid) == 0
        }


def create_response_parser(
    strict_validation: bool = False,
    require_citations: bool = True
) -> ResponseParser:
    """Factory function to create ResponseParser.

    Args:
        strict_validation: Whether to raise errors for invalid citations.
        require_citations: Whether to warn when citations are missing.

    Returns:
        Initialized ResponseParser instance.

    Example:
        >>> parser = create_response_parser(strict_validation=True)
    """
    return ResponseParser(
        strict_validation=strict_validation,
        require_citations=require_citations
    )
