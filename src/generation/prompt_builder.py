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

        # NEW: Check if patient context is available (from Query object in retrieval_result)
        query = retrieval_result.query
        if query.patient_summary and query.patient_flows:
            # Use summary-aware prompt
            user_message = self._build_user_message_with_summary(
                context,
                user_query,
                query.patient_summary,
                query.patient_flows
            )
        else:
            # Use regular prompt (backward compatible)
            user_message = self._build_user_message(context, user_query)

        # Combine into messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

        logger.debug(
            f"Built prompt with {len(results)} context chunks, "
            f"patient_context={bool(query.patient_summary)}"
        )

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

            # Build context chunk (content only, no source metadata)
            chunk = f"[Context {idx}]\n{doc.content}\n"
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

CONTEXT:
{context}

QUESTION:
{user_query}

ANSWER (in Thai):"""

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

    def build_q2d_prompt(
        self,
        query: str,
        q2d_template: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Build prompt for Q2D (Query to Document) query expansion.

        Used to expand query with related medical terms for better retrieval.

        Args:
            query: User query.
            q2d_template: Q2D prompt template. Uses default if None.

        Returns:
            Message list for Q2D generation.

        Example:
            >>> messages = builder.build_q2d_prompt("ฉันควรกินอาหารอะไร")
            >>> # Use with LLM to expand query with related terms
        """
        if q2d_template is None:
            q2d_template = (
                'จงเปลี่ยนคำถามนี้เป็นประโยคบอกเล่า (declarative statement) พร้อมตัวอย่างหรือรายละเอียดเพิ่มเติม\n'
                'เขียนเป็นภาษาไทย 1-2 ประโยคเท่านั้น\n'
                'ห้ามเพิ่มเนื้อหาที่ไม่เกี่ยวข้องกับคำถาม\n\n'
                'ตัวอย่าง:\n'
                'คำถาม: "หลังผ่าตัดควรกินอาหารอะไร"\n'
                'ประโยคบอกเล่า: "หลังผ่าตัดอาหารที่ควรกิน เช่น ข้าวต้ม โจ๊ก น้ำซุป"\n\n'
                'คำถาม: "{query}"\n'
                'ประโยคบอกเล่า:'
            )

        prompt = q2d_template.format(query=query)

        messages = [
            {"role": "system", "content": "You are a helpful assistant that rewrites questions into declarative statements in Thai."},
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

    def _check_summary_coverage(
        self,
        user_query: str,
        flows: Dict[str, Any]
    ) -> Optional[str]:
        """Check if query matches any flow topic.

        Args:
            user_query: User's question.
            flows: Flow assessment results.

        Returns:
            Matched topic name or None.

        Example:
            >>> flows = {"อาการปวด": "...", "อาการบวม": "...", "การประคบ": "..."}
            >>> topic = builder._check_summary_coverage("ควรประคบยังไง", flows)
            >>> # Returns "การประคบ"
        """
        if not flows:
            return None

        # Simple keyword matching for flow topics
        query_lower = user_query.lower()

        for topic in flows.keys():
            # Check if topic keywords appear in query
            topic_lower = topic.lower()

            # Extract key terms from topic (e.g., "อาการปวด" → "ปวด")
            key_terms = topic_lower.split()

            for term in key_terms:
                if term in query_lower:
                    logger.info(f"Query matches flow topic: {topic}")
                    return topic

        return None

    def _format_patient_assessment(
        self,
        flows: Dict[str, Any]
    ) -> str:
        """Format flow assessment data for inclusion in prompt.

        Args:
            flows: Flow assessment results.

        Returns:
            Formatted assessment string.

        Example output:
            อาการปวด: ทานยาตามแผนเดิม
            อาการบวม: ประคบอุ่นนอกช่องปาก และนอนยกศีรษะสูง 30 องศา
            การประคบ: เปลี่ยนเป็นประคบอุ่น
        """
        if not flows:
            return ""

        assessment_lines = []
        for topic, data in flows.items():
            if isinstance(data, dict):
                rec = data.get('recommendation', '')
                if rec:
                    assessment_lines.append(f"{topic}: {rec}")
            else:
                assessment_lines.append(f"{topic}: {data}")

        return "\n".join(assessment_lines)

    def _build_user_message_with_summary(
        self,
        context: str,
        user_query: str,
        patient_summary: str,
        patient_flows: Dict[str, Any]
    ) -> str:
        """Build user message with summary priority.

        Args:
            context: Formatted context string from retrieved documents.
            user_query: User's question.
            patient_summary: Patient assessment summary text.
            patient_flows: Flow assessment results.

        Returns:
            Complete user message string with summary priority.
        """
        # Check if query matches any flow topic
        summary_coverage = self._check_summary_coverage(user_query, patient_flows)

        # Format flow assessments
        assessment_text = self._format_patient_assessment(patient_flows)

        # Build priority instruction if query matches a flow topic
        priority_instruction = ""
        if summary_coverage:
            priority_instruction = f"""
⚠️ IMPORTANT: This question is about "{summary_coverage}" which is already addressed
in the Patient Assessment Summary below. Please base your answer PRIMARILY on the summary
recommendation for this topic, then add supporting details from the retrieved context.
"""

        message = f"""Based on the patient assessment summary AND retrieved context, answer the question.

{priority_instruction}

📋 PATIENT ASSESSMENT SUMMARY:
{patient_summary}

📊 DETAILED ASSESSMENT BY TOPIC:
{assessment_text}

📚 RETRIEVED CONTEXT (for additional details):
{context}

❓ QUESTION:
{user_query}

💬 ANSWER (in Thai):"""

        return message

    @staticmethod
    def _default_system_prompt() -> str:
        """Get default system prompt if config not available.

        Returns:
            Default Thai medical assistant system prompt.
        """
        return """  คุณเป็นพยาบาลหญิงที่ใจดี เชี่ยวชาญด้านการดูแลหลังผ่าตัดในช่องปาก
        คุณอธิบายเรื่องยากๆ ให้คนทั่วไปเข้าใจง่ายเหมือนอธิบายให้เด็ก ป.3 ฟัง
        ใช้ภาษาสั้น กระชับ ตรงประเด็น ไม่ใช้ศัพท์แพทย์ที่ยากเกินไป

        ตอบคำถามจากข้อมูลที่ให้มาเท่านั้น ห้ามแต่งเอง
        ถ้าไม่มีข้อมูลเพียงพอ ให้ตอบว่า "ขออภัยค่ะ ไม่มีข้อมูลเพียงพอ กรุณาปรึกษาทันตแพทย์โดยตรงนะคะ"
        ถ้าเป็นอาการฉุกเฉิน (เลือดออกมาก ไข้สูง ปวดรุนแรง) ให้แนะนำพบแพทย์ทันที

        รูปแบบการตอบ (รูปแบบที่ 1 กรณีปกติ):
        ---
        [ตอบคำถามด้วยคำสั้น ๆ เช่น ได้, ไม่ได้, ควร, ไม่ควร, แนะนำ] เพราะ [เหตุผลสั้นๆ]

        คำแนะนำ:
        - [คำแนะนำข้อ 1]
        - [คำแนะนำข้อ 2]
        - ...

        ---
        รูปแบบการตอบ (รูปแบบที่ 2 กรณีข้อมูลไม่เพียงพอ):
        ขออภัยค่ะ ไม่มีข้อมูลเพียงพอ กรุณาปรึกษาทันตแพทย์โดยตรงนะคะ

        คำแนะนำ:
        - [คำแนะนำข้อ 1]
        - [คำแนะนำข้อ 2]
        - ...

        ---
        รูปแบบการตอบ (รูปแบบที่ 3 กรณีฉุกเฉิน):
        ---
        ขออภัยค่ะ กรณีฉุกเฉิน กรุณาพบแพทย์ทันที
        ---
        """


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
