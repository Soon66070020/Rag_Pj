"""Response generation module for RAG-based medical Q&A.

This module handles the complete generation pipeline:
1. DeepSeek API client with retry logic
2. Prompt construction with context and citations
3. Response parsing and citation validation
4. Main generation engine orchestrating all components

All components enforce medical safety constraints:
- Never invents information not in context
- Requires citations for factual statements
- Strips PII from queries before API calls
- Returns fallback message when context is insufficient

Example:
    >>> from src.generation import create_generation_engine
    >>> from config.settings import get_settings
    >>>
    >>> settings = get_settings()
    >>> engine = create_generation_engine(
    ...     settings.model_config['llm'],
    ...     settings.prompts_config,
    ...     api_key="sk-xxx"
    ... )
    >>>
    >>> # Generate response
    >>> from src.retrieval import retrieve
    >>> retrieval_result = retrieve("ฉันควรกินยาแก้ปวดไหม")
    >>> response = engine.generate(retrieval_result)
    >>>
    >>> print(response.answer)
    >>> print(f"Citations: {response.citations}")
"""

from src.generation.client import DeepSeekClient, create_client_from_config
from src.generation.prompt_builder import PromptBuilder, create_prompt_builder
from src.generation.response_parser import ResponseParser, create_response_parser
from src.generation.engine import GenerationEngine, create_generation_engine


__all__ = [
    # Main classes
    'DeepSeekClient',
    'PromptBuilder',
    'ResponseParser',
    'GenerationEngine',

    # Factory functions
    'create_client_from_config',
    'create_prompt_builder',
    'create_response_parser',
    'create_generation_engine',
]
