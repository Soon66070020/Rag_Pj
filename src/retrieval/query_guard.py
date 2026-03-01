"""Multi-LLM Query Guard Pipeline.

Implements a 3-stage LLM pipeline for query preprocessing:
1. Scope Check (LLM1): Is the query about dental post-op care?
2. Clarification + Rewrite (LLM2): Is the query clear enough? If not, rewrite using patient context.
3. Classification (LLM3): Assign multi-category labels for retrieval filtering.

Uses DeepSeek-chat with JSON mode for structured output.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from src.core.exceptions import QueryGuardError
from src.generation.client import DeepSeekClient


logger = logging.getLogger(__name__)

# Fallback message when query is out of domain
OUT_OF_DOMAIN_FALLBACK = (
    "ขออภัยค่ะ ระบบนี้ตอบได้เฉพาะคำถามเกี่ยวกับการดูแลหลังผ่าตัดในช่องปากเท่านั้นค่ะ "
    "หากมีคำถามเกี่ยวกับอาการหลังผ่าตัด ยา อาหาร หรือการดูแลแผล สามารถถามได้เลยนะคะ"
)


@dataclass
class ScopeCheckResult:
    """Result from LLM1 scope check."""
    is_in_scope: bool
    reasoning: str


@dataclass
class ClarificationResult:
    """Result from LLM2 clarification + rewrite."""
    clarification_needed: bool
    reason: str
    rewritten_query: str


@dataclass
class CategoryAssignment:
    """Single category assignment with confidence."""
    category: str
    confidence: float


@dataclass
class ClassificationResult:
    """Result from LLM3 multi-category classification."""
    categories: List[CategoryAssignment]
    reasoning: str


@dataclass
class QueryGuardResult:
    """Complete result from the 3-stage guard pipeline."""
    is_in_scope: bool
    clarification_needed: bool
    original_query: str
    effective_query: str
    classification: Optional[ClassificationResult] = None
    fallback_message: Optional[str] = None
    guard_time_ms: float = 0.0


class QueryGuard:
    """3-stage LLM query guard pipeline.

    Stages:
        1. Scope Check: Determine if query is about dental post-op care.
        2. Clarification + Rewrite: Check clarity, rewrite with patient context if needed.
        3. Classification: Assign categories for retrieval filtering.

    All stages use DeepSeek-chat with JSON mode.
    Fail-open: If any LLM call fails, the query proceeds (is_in_scope=True, no rewrite).
    """

    CONFIDENCE_THRESHOLD = 0.4

    def __init__(self, llm_client: DeepSeekClient, prompts: Dict[str, Any]):
        self.llm_client = llm_client
        self.prompts = prompts

    def run(
        self,
        query: str,
        patient_context: Optional[Dict[str, Any]] = None
    ) -> QueryGuardResult:
        """Run the full 3-stage guard pipeline.

        Args:
            query: User query text.
            patient_context: Optional patient data (procedures, assessment, flows).

        Returns:
            QueryGuardResult with scope, clarification, and classification info.
        """
        start_time = time.time()
        effective_query = query

        # Stage 1: Scope Check
        scope_result = self.check_scope(query)
        if not scope_result.is_in_scope:
            logger.info(f"Query out of domain: {scope_result.reasoning}")
            return QueryGuardResult(
                is_in_scope=False,
                clarification_needed=False,
                original_query=query,
                effective_query=query,
                fallback_message=OUT_OF_DOMAIN_FALLBACK,
                guard_time_ms=(time.time() - start_time) * 1000,
            )

        # Stage 2: Clarification + Rewrite
        clarification_result = self.check_clarification(query, patient_context)
        clarification_needed = clarification_result.clarification_needed
        if clarification_needed and clarification_result.rewritten_query:
            effective_query = clarification_result.rewritten_query
            logger.info(f"Query rewritten: '{query}' -> '{effective_query}'")

        # Stage 3: Classification
        classification_result = self.classify_query(effective_query)

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Guard pipeline complete in {elapsed_ms:.0f}ms: "
            f"scope=True, clarification={clarification_needed}, "
            f"categories={[c.category for c in classification_result.categories]}"
        )

        return QueryGuardResult(
            is_in_scope=True,
            clarification_needed=clarification_needed,
            original_query=query,
            effective_query=effective_query,
            classification=classification_result,
            guard_time_ms=elapsed_ms,
        )

    def check_scope(self, query: str) -> ScopeCheckResult:
        """LLM1: Check if query is within dental post-op scope."""
        prompt_template = self.prompts.get('scope_check_prompt', '')
        if not prompt_template:
            logger.warning("scope_check_prompt not found, defaulting to in-scope")
            return ScopeCheckResult(is_in_scope=True, reasoning="no prompt configured")

        user_message = prompt_template.replace("{query}", query)

        try:
            response = self.llm_client.generate(
                messages=[{"role": "user", "content": user_message}],
                response_format={"type": "json_object"},
            )
            data = self._parse_json(response.get("content", ""), {
                "is_in_scope": True,
                "reasoning": "parse error fallback",
            })
            return ScopeCheckResult(
                is_in_scope=bool(data.get("is_in_scope", True)),
                reasoning=str(data.get("reasoning", "")),
            )
        except Exception as e:
            logger.error(f"Scope check failed (fail-open): {e}")
            return ScopeCheckResult(is_in_scope=True, reasoning=f"error: {e}")

    def check_clarification(
        self,
        query: str,
        patient_context: Optional[Dict[str, Any]] = None,
    ) -> ClarificationResult:
        """LLM2: Check if query needs clarification, rewrite if needed."""
        prompt_template = self.prompts.get('clarification_check_prompt', '')
        if not prompt_template:
            logger.warning("clarification_check_prompt not found, skipping")
            return ClarificationResult(
                clarification_needed=False, reason="", rewritten_query=""
            )

        context_str = self._build_patient_context_str(patient_context)
        user_message = (
            prompt_template
            .replace("{query}", query)
            .replace("{patient_context}", context_str)
        )

        try:
            response = self.llm_client.generate(
                messages=[{"role": "user", "content": user_message}],
                response_format={"type": "json_object"},
            )
            data = self._parse_json(response.get("content", ""), {
                "clarification_needed": False,
                "reason": "",
                "rewritten_query": "",
            })
            return ClarificationResult(
                clarification_needed=bool(data.get("clarification_needed", False)),
                reason=str(data.get("reason", "")),
                rewritten_query=str(data.get("rewritten_query", "")),
            )
        except Exception as e:
            logger.error(f"Clarification check failed (fail-open): {e}")
            return ClarificationResult(
                clarification_needed=False, reason=f"error: {e}", rewritten_query=""
            )

    def classify_query(self, query: str) -> ClassificationResult:
        """LLM3: Multi-category classification."""
        prompt_template = self.prompts.get('classification_prompt', '')
        if not prompt_template:
            logger.warning("classification_prompt not found, defaulting Post-op Care")
            return ClassificationResult(
                categories=[CategoryAssignment(category="Post-op Care", confidence=0.5)],
                reasoning="no prompt configured",
            )

        user_message = prompt_template.replace("{query}", query)

        try:
            response = self.llm_client.generate(
                messages=[{"role": "user", "content": user_message}],
                response_format={"type": "json_object"},
            )
            data = self._parse_json(response.get("content", ""), {
                "categories": [{"category": "Post-op Care", "confidence": 0.5}],
                "reasoning": "parse error fallback",
            })

            raw_categories = data.get("categories", [])
            categories = []
            for cat in raw_categories:
                confidence = float(cat.get("confidence", 0.0))
                if confidence >= self.CONFIDENCE_THRESHOLD:
                    categories.append(CategoryAssignment(
                        category=str(cat.get("category", "Post-op Care")),
                        confidence=confidence,
                    ))

            if not categories:
                categories = [CategoryAssignment(category="Post-op Care", confidence=0.5)]

            # Sort by confidence descending
            categories.sort(key=lambda c: c.confidence, reverse=True)

            return ClassificationResult(
                categories=categories,
                reasoning=str(data.get("reasoning", "")),
            )
        except Exception as e:
            logger.error(f"Classification failed (fail-open): {e}")
            return ClassificationResult(
                categories=[CategoryAssignment(category="Post-op Care", confidence=0.5)],
                reasoning=f"error: {e}",
            )

    def _parse_json(self, content: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON from LLM response, return fallback on failure."""
        if not content or not content.strip():
            logger.warning("Empty LLM response, using fallback")
            return fallback
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}, content: {content[:200]}")
            return fallback

    def _build_patient_context_str(
        self, patient_context: Optional[Dict[str, Any]]
    ) -> str:
        """Build a text summary of patient context for the clarification prompt."""
        if not patient_context:
            return "ไม่มีข้อมูลบริบทผู้ป่วย"

        parts = []

        # Extract procedures
        patient = patient_context.get("patient", {})
        basic_info = patient.get("basic_info", {})
        procedures = basic_info.get("procedures", [])
        if procedures:
            parts.append(f"หัตถการที่ทำ: {', '.join(str(p) for p in procedures)}")

        # Extract assessment data
        assessment = patient.get("assessment_data", {})
        if assessment:
            pain_desc = assessment.get("pain_description", "")
            if pain_desc:
                parts.append(f"อาการปวด: {pain_desc}")

            swelling = assessment.get("swelling_description", "")
            if swelling:
                parts.append(f"อาการบวม: {swelling}")

            bleeding = assessment.get("bleeding_status", "")
            if bleeding:
                parts.append(f"เลือด: {bleeding}")

            numbness = assessment.get("numbness_description", "")
            if numbness:
                parts.append(f"อาการชา: {numbness}")

            compress = assessment.get("compress_type", "")
            if compress:
                parts.append(f"การประคบ: {compress}")

        # Extract summary
        summary = patient_context.get("summary", {})
        summary_text = summary.get("summary", "") if isinstance(summary, dict) else str(summary)
        if summary_text:
            parts.append(f"สรุปผลประเมิน: {summary_text}")

        # Extract flows (recommendations)
        flows = patient_context.get("flows", {})
        if flows:
            flow_parts = []
            for topic, data in flows.items():
                rec = data.get("recommendation", "") if isinstance(data, dict) else str(data)
                if rec:
                    flow_parts.append(f"- {topic}: {rec}")
            if flow_parts:
                parts.append("คำแนะนำจากผลประเมิน:\n" + "\n".join(flow_parts))

        return "\n".join(parts) if parts else "ไม่มีข้อมูลบริบทผู้ป่วย"
