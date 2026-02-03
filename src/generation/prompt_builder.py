"""Prompt builder for medical RAG system.

This module constructs prompts for the LLM by combining:
1. System instructions from prompts.yaml
2. Retrieved context with source citations
3. User query

All prompts enforce citation requirements for medical accuracy.
"""

import logging
from typing import List, Dict, Any, Optional

from src.core.types import RetrievalResult, SearchResult, Document


logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builder for constructing LLM prompts with retrieved context.

    Formats prompts for Thai medical Q&A with mandatory citation requirements.
    Uses templates from prompts.yaml configuration.

    Attributes:
        system_prompt: Base system instruction template.
        empty_context_prompt: Fallback for when no context is found.
        citation_instruction: Citation format requirements.
        max_context_length: Maximum tokens for context section.

    Example:
        >>> builder = PromptBuilder(prompts_config)
        >>> messages = builder.build_prompt(retrieval_result, "ฉันควรกินยาแก้ปวดไหม")
        >>> # Returns list of message dicts ready for LLM
    """

    def __init__(
        self,
        prompts_config: Dict[str, Any],
        max_context_length: int = 4000
    ):
        """Initialize prompt builder.

        Args:
            prompts_config: Prompts configuration from prompts.yaml.
            max_context_length: Maximum characters for context section.
        """
        self.system_prompt = prompts_config.get('system_prompt', '')
        self.empty_context_prompt = prompts_config.get('empty_context_prompt', '')
        self.citation_instruction = prompts_config.get('citation_format_instruction', '')
        self.max_context_length = max_context_length

        if not self.system_prompt:
            logger.warning("System prompt not found in config, using default")
            self.system_prompt = self._default_system_prompt()

        logger.info("PromptBuilder initialized")

    def build_prompt(
        self,
        retrieval_result: RetrievalResult,
        user_query: str,
        include_all_candidates: bool = False
    ) -> List[Dict[str, str]]:
        """Build complete prompt messages for LLM.

        Args:
            retrieval_result: Results from retrieval pipeline.
            user_query: Original user question.
            include_all_candidates: If True, use all candidates instead of just reranked.

        Returns:
            List of message dicts in OpenAI format:
                [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

        Example:
            >>> messages = builder.build_prompt(retrieval_result, "ยาแก้ปวดคืออะไร")
            >>> # Returns formatted messages ready for API call
        """
        # Choose which results to use
        results = (
            retrieval_result.candidates if include_all_candidates
            else retrieval_result.reranked_results
        )

        # Check if we have context
        if not results:
            logger.warning("No context available, using empty context prompt")
            return self._build_empty_context_prompt(user_query)

        # Build context section
        context = self._format_context(results)

        # Build user message with context + query
        user_message = self._build_user_message(context, user_query)

        # Combine into messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

        logger.debug(f"Built prompt with {len(results)} context chunks")

        return messages

    def _format_context(self, results: List[SearchResult]) -> str:
        """Format retrieved documents into context string.

        Args:
            results: List of search results to format.

        Returns:
            Formatted context string with source citations.

        Example output:
            ---
            [Context 1] (Source: dental_care.pdf, Page: 5, Score: 0.92)
            การดูแลหลังผ่าตัดฟัน ควรรับประทานยาแก้ปวดตามแพทย์สั่ง...

            [Context 2] (Source: medication.pdf, Page: 12, Score: 0.87)
            ยาแก้ปวดที่แนะนำคือ พาราเซตามอล 500 มก. ทุก 4-6 ชั่วโมง...
            ---
        """
        context_parts = []

        for idx, result in enumerate(results, start=1):
            doc = result.document

            # Format source information
            source_info = (
                f"(Source: {doc.source_file}, "
                f"Page: {doc.page_number}, "
                f"Score: {result.score:.2f})"
            )

            # Add category if available
            if doc.category:
                source_info = (
                    f"(Category: {doc.category}, "
                    f"Source: {doc.source_file}, "
                    f"Page: {doc.page_number}, "
                    f"Score: {result.score:.2f})"
                )

            # Build context chunk
            chunk = f"[Context {idx}] {source_info}\n{doc.content}\n"
            context_parts.append(chunk)

            # Check if we're exceeding max length
            current_length = sum(len(part) for part in context_parts)
            if current_length > self.max_context_length:
                logger.warning(
                    f"Context truncated at {idx}/{len(results)} chunks "
                    f"to stay within {self.max_context_length} chars"
                )
                break

        # Join all context chunks
        formatted_context = "---\n" + "\n".join(context_parts) + "---\n"

        return formatted_context

    def _build_user_message(self, context: str, user_query: str) -> str:
        """Build the user message combining context and query.

        Args:
            context: Formatted context string.
            user_query: User's question.

        Returns:
            Complete user message string.
        """
        message = f"""Based on the following context, please answer the question.

{self.citation_instruction}

CONTEXT:
{context}

QUESTION:
{user_query}

ANSWER (in Thai with citations):"""

        return message

    def _build_empty_context_prompt(self, user_query: str) -> List[Dict[str, str]]:
        """Build prompt for when no context is available.

        Args:
            user_query: User's question.

        Returns:
            Message list with fallback instruction.
        """
        messages = [
            {"role": "system", "content": self.empty_context_prompt},
            {"role": "user", "content": user_query}
        ]

        return messages

    def build_hyde_prompt(
        self,
        query: str,
        hyde_template: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Build prompt for HyDE (Hypothetical Document Embeddings).

        Used to generate hypothetical answers for query expansion.

        Args:
            query: User query.
            hyde_template: HyDE prompt template. Uses default if None.

        Returns:
            Message list for HyDE generation.

        Example:
            >>> messages = builder.build_hyde_prompt("ฉันควรกินอาหารอะไร")
            >>> # Use with LLM to generate hypothetical answer
        """
        if hyde_template is None:
            hyde_template = (
                'Given this question from a post-oral surgery patient:\n'
                '"{query}"\n\n'
                'Generate a hypothetical detailed answer (2-3 sentences) that a Thai '
                'medical professional might provide.\n'
                'Answer in Thai language with polite tone (ครับ/ค่ะ).'
            )

        prompt = hyde_template.format(query=query)

        messages = [
            {"role": "system", "content": "You are a Thai medical professional."},
            {"role": "user", "content": prompt}
        ]

        return messages

    def extract_context_sources(
        self,
        retrieval_result: RetrievalResult
    ) -> List[Dict[str, Any]]:
        """Extract source information from retrieval result.

        Useful for citation validation.

        Args:
            retrieval_result: Retrieval pipeline result.

        Returns:
            List of dicts with source metadata:
                [{"source_file": "...", "page_number": ..., "content": "..."}, ...]

        Example:
            >>> sources = builder.extract_context_sources(result)
            >>> for src in sources:
            ...     print(f"{src['source_file']}:{src['page_number']}")
        """
        sources = []

        for result in retrieval_result.reranked_results:
            doc = result.document
            sources.append({
                "source_file": doc.source_file,
                "page_number": doc.page_number,
                "category": doc.category,
                "content": doc.content,
                "score": result.score
            })

        return sources

    def validate_context_quality(
        self,
        retrieval_result: RetrievalResult,
        min_score: float = 0.5,
        min_results: int = 1
    ) -> bool:
        """Check if retrieved context is sufficient for generation.

        Args:
            retrieval_result: Retrieval result to validate.
            min_score: Minimum acceptable relevance score.
            min_results: Minimum number of results required.

        Returns:
            True if context quality is acceptable.

        Example:
            >>> if builder.validate_context_quality(result, min_score=0.6):
            ...     # Proceed with generation
            ... else:
            ...     # Return "insufficient information" message
        """
        results = retrieval_result.reranked_results

        # Check if we have enough results
        if len(results) < min_results:
            logger.warning(f"Only {len(results)} results, need at least {min_results}")
            return False

        # Check if top result meets minimum score
        if results and results[0].score < min_score:
            logger.warning(
                f"Top result score {results[0].score:.2f} below threshold {min_score}"
            )
            return False

        return True

    @staticmethod
    def _default_system_prompt() -> str:
        """Get default system prompt if config not available.

        Returns:
            Default Thai medical assistant system prompt.
        """
        return """You are an expert dental assistant AI specializing in post-oral surgery care.

Your task is to answer patient questions based ONLY on the provided context.

CRITICAL RULES:
1. **Answer in Thai language** using a polite and professional tone (ครับ/ค่ะ).
2. **Do NOT use outside knowledge**. If the answer is not in the context, say "ขออภัยครับ/ค่ะ ไม่มีข้อมูลเพียงพอในระบบ"
3. **MANDATORY CITATION**: Every factual statement MUST include a citation in this format: [Source: filename, Page: number]
4. If information comes from multiple sources, cite all.
5. Be specific and actionable. Focus on practical advice.
6. If emergency symptoms are mentioned, advise immediate medical consultation: "กรุณาพบทันตแพทย์โดยด่วน"

Remember: Citations are MANDATORY. Do not skip them."""


def create_prompt_builder(prompts_config: Dict[str, Any]) -> PromptBuilder:
    """Factory function to create PromptBuilder from config.

    Args:
        prompts_config: Configuration dict from prompts.yaml.

    Returns:
        Initialized PromptBuilder instance.

    Example:
        >>> from config.settings import get_settings
        >>> settings = get_settings()
        >>> builder = create_prompt_builder(settings.prompts_config)
    """
    return PromptBuilder(prompts_config)
