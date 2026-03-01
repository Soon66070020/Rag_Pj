"""FastAPI service wrapping the RAG pipeline.

Endpoints:
- POST /generate : Full RAG pipeline (retrieve + generate answer)
- GET  /health   : Service readiness check

Usage:
    conda activate env_rag
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any

# Add project root to path so config/src imports work
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, HTTPException

from api.models import PatientRequest, GenerateResponse, HealthResponse, GuardInfo, CategoryInfo

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# Global service instances (initialized on startup)
retrieval_service = None
generation_engine = None


def _preprocess_flows(flows: Dict[str, Any]) -> Dict[str, str]:
    """Extract recommendation strings from nested flow dicts.

    API sends flows as:
        {"topic": {"risk_level": "...", "recommendation": "...", "reason": "..."}}
    Pipeline expects:
        {"topic": "recommendation text"}
    """
    processed = {}
    for topic, data in flows.items():
        if isinstance(data, dict):
            rec = data.get("recommendation", "")
            if rec:
                processed[topic] = rec
        else:
            processed[topic] = str(data)
    return processed


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global retrieval_service, generation_engine

    logger.info("Loading RAG pipeline models...")
    start = time.time()

    from config.settings import get_settings
    from src.retrieval.service import RetrievalService
    from src.generation.engine import create_generation_engine

    settings = get_settings()

    # Initialize retrieval service (loads BGE-M3 + Reranker)
    retrieval_service = RetrievalService(settings)

    # Initialize generation engine (DeepSeek LLM client)
    generation_engine = create_generation_engine(
        llm_config=settings.model_config["llm"],
        prompts_config=settings.prompts,
        api_key=settings.deepseek_api_key,
    )

    elapsed = time.time() - start
    logger.info(f"RAG pipeline ready in {elapsed:.1f}s")

    yield

    # Cleanup
    logger.info("Shutting down RAG service")


app = FastAPI(
    title="Dental Post-Op RAG API",
    description="RAG-based Q&A for dental post-operative care",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check service readiness."""
    return HealthResponse(
        status="ok" if retrieval_service is not None else "loading",
        models_loaded=retrieval_service is not None,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: PatientRequest):
    """Full RAG pipeline: retrieve relevant docs + generate answer.

    Input: Patient JSON (same format as examples/API_input_ex.json)
    - Query extracted from patient.assessment_data.additional_questions
    - Patient context from flows + summary

    Output: JSON with answer, citations, and metrics
    """
    if retrieval_service is None or generation_engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    # 1. Extract query
    assessment = request.patient.get("assessment_data", {})
    query_text = assessment.get("additional_questions", "")
    if not query_text:
        raise HTTPException(
            status_code=400,
            detail="No question found in patient.assessment_data.additional_questions",
        )

    # 2. Build patient context for guard pipeline (includes full patient data)
    full_context = {
        "patient": request.patient,
        "flows": request.flows,
        "summary": request.summary,
    }

    try:
        # 3. Retrieve with guard pipeline
        guard_result, retrieval_result = retrieval_service.retrieve_with_guard(
            query_text, patient_context=full_context
        )

        # Build GuardInfo for response
        guard_info = GuardInfo(
            is_in_scope=guard_result.is_in_scope,
            clarification_needed=guard_result.clarification_needed,
            original_query=guard_result.original_query,
            effective_query=guard_result.effective_query,
            classification=[
                CategoryInfo(category=c.category, confidence=c.confidence)
                for c in guard_result.classification.categories
            ] if guard_result.classification else None,
            guard_time_ms=guard_result.guard_time_ms,
        )

        # Out of domain → return fallback answer immediately
        if not guard_result.is_in_scope:
            return GenerateResponse(
                answer=guard_result.fallback_message or "ขออภัย คำถามนี้อยู่นอกขอบเขตของระบบ",
                citations=[],
                query=query_text,
                retrieval_time_ms=0.0,
                generation_time_ms=0.0,
                documents_used=0,
                guard=guard_info,
            )

        # 4. Generate answer using effective_query
        effective_query = guard_result.effective_query
        response = generation_engine.generate(
            retrieval_result=retrieval_result,
            user_query=effective_query,
        )

        return GenerateResponse(
            answer=response.answer,
            citations=response.citations,
            query=effective_query,
            retrieval_time_ms=retrieval_result.retrieval_time_ms,
            generation_time_ms=response.generation_time_ms,
            documents_used=len(response.context_used),
            guard=guard_info,
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
