"""Pydantic models for the RAG API request/response schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class PatientRequest(BaseModel):
    """Request body: full patient JSON (same format as examples/API_input_ex.json).

    Accepts the entire patient assessment payload. Key fields used:
    - patient.assessment_data.additional_questions -> query
    - flows -> patient context (recommendations)
    - summary -> patient summary (overall risk)
    """
    patient: Dict[str, Any]
    flows: Dict[str, Any]
    summary: Dict[str, Any]
    language: Optional[str] = "th"
    errors: Optional[Any] = None


class CategoryInfo(BaseModel):
    """Single category assignment from LLM3 classification."""
    category: str
    confidence: float


class GuardInfo(BaseModel):
    """Query guard pipeline results included in every response."""
    is_in_scope: bool
    clarification_needed: bool
    original_query: str
    effective_query: str
    classification: Optional[List[CategoryInfo]] = None
    guard_time_ms: float


class GenerateResponse(BaseModel):
    """Response body for /generate endpoint."""
    answer: str
    citations: List[str]
    query: str
    retrieval_time_ms: float
    generation_time_ms: float
    documents_used: int
    guard: Optional[GuardInfo] = None


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    status: str
    models_loaded: bool
