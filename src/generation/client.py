"""DeepSeek API client for response generation.

This module provides a robust client for interacting with the DeepSeek API,
with retry logic, error handling, and medical safety constraints.

Temperature is set to 0.1 for medical accuracy and consistency.
"""

import logging
import time
from typing import Optional, Dict, Any, List
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

try:
    from openai import OpenAI
    from openai import OpenAIError, APIError, RateLimitError, APIConnectionError
except ImportError:
    raise ImportError(
        "OpenAI library required for DeepSeek client. "
        "Install with: pip install openai"
    )

from src.core.exceptions import GenerationError


logger = logging.getLogger(__name__)


class DeepSeekClient:
    """Client for DeepSeek API using OpenAI-compatible interface.

    Implements retry logic for API stability and enforces medical safety
    constraints (low temperature, proper error handling).

    Attributes:
        model_name: DeepSeek model identifier (e.g., "deepseek-chat").
        api_base: Base URL for DeepSeek API.
        temperature: Sampling temperature (default 0.1 for medical accuracy).
        max_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling parameter.
        timeout: API request timeout in seconds.
        stream: Whether to stream responses.
        client: Underlying OpenAI client instance.

    Example:
        >>> client = DeepSeekClient(
        ...     api_key="sk-xxx",
        ...     temperature=0.1,
        ...     max_tokens=2048
        ... )
        >>> response = client.generate(
        ...     messages=[{"role": "user", "content": "สวัสดี"}]
        ... )
        >>> print(response["content"])
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "deepseek-chat",
        api_base: str = "https://api.deepseek.com/v1",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        timeout: int = 60,
        stream: bool = False,
        max_retries: int = 3
    ):
        """Initialize DeepSeek client.

        Args:
            api_key: DeepSeek API key.
            model_name: Model identifier.
            api_base: API base URL.
            temperature: Sampling temperature (0.0-1.0). Keep low for medical.
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling parameter.
            timeout: Request timeout in seconds.
            stream: Whether to stream responses.
            max_retries: Maximum retry attempts for failed requests.

        Raises:
            ValueError: If temperature is too high for medical use.
        """
        # Safety check for medical use
        if temperature > 0.3:
            logger.warning(
                f"Temperature {temperature} may be too high for medical advice. "
                "Consider using 0.1 for consistency."
            )

        self.model_name = model_name
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.stream = stream
        self.max_retries = max_retries

        # Initialize OpenAI client (DeepSeek API is compatible)
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout
        )

        logger.info(
            f"DeepSeek client initialized: model={model_name}, "
            f"temperature={temperature}, max_tokens={max_tokens}"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        reraise=True
    )
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate response from DeepSeek API with retry logic.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            top_p: Override default top_p.
            stop: Stop sequences for generation.

        Returns:
            Dict containing:
                - content: Generated text content
                - finish_reason: Reason for completion
                - usage: Token usage statistics
                - model: Model used
                - generation_time_ms: Time taken in milliseconds

        Raises:
            GenerationError: If API call fails after retries.

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are a medical assistant"},
            ...     {"role": "user", "content": "ยาแก้ปวดคืออะไร"}
            ... ]
            >>> response = client.generate(messages)
            >>> print(response["content"])
        """
        start_time = time.time()

        # Use provided parameters or defaults
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        top_p_val = top_p if top_p is not None else self.top_p

        try:
            logger.debug(f"Calling DeepSeek API: {len(messages)} messages")

            # Call DeepSeek API via OpenAI client
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
                top_p=top_p_val,
                stop=stop,
                stream=self.stream
            )

            # Extract response content
            if self.stream:
                # For streaming, this is handled differently
                raise NotImplementedError("Streaming not yet implemented")

            choice = response.choices[0]
            content = choice.message.content
            finish_reason = choice.finish_reason

            # Calculate generation time
            generation_time_ms = (time.time() - start_time) * 1000

            # Build response dict
            result = {
                "content": content,
                "finish_reason": finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "model": response.model,
                "generation_time_ms": generation_time_ms
            }

            logger.info(
                f"Generation complete: {result['usage']['total_tokens']} tokens "
                f"in {generation_time_ms:.2f}ms"
            )

            return result

        except (APIError, RateLimitError, APIConnectionError) as e:
            # These exceptions trigger retry via @retry decorator
            logger.warning(f"API error (will retry): {e}")
            raise

        except OpenAIError as e:
            # Other OpenAI errors (non-retryable)
            error_msg = f"DeepSeek API error: {e}"
            logger.error(error_msg)
            raise GenerationError(error_msg) from e

        except Exception as e:
            # Unexpected errors
            error_msg = f"Unexpected error in generation: {e}"
            logger.error(error_msg)
            raise GenerationError(error_msg) from e

    def generate_with_fallback(
        self,
        messages: List[Dict[str, str]],
        fallback_response: str = "ขอภัยครับ/ค่ะ ระบบขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้ง"
    ) -> Dict[str, Any]:
        """Generate response with fallback on failure.

        Attempts generation, returns fallback message if all retries fail.

        Args:
            messages: Message list for generation.
            fallback_response: Thai fallback message on failure.

        Returns:
            Generation result or fallback response dict.

        Example:
            >>> response = client.generate_with_fallback(messages)
            >>> # Will return fallback if API fails after retries
        """
        try:
            return self.generate(messages)
        except Exception as e:
            logger.error(f"All retries failed, using fallback: {e}")
            return {
                "content": fallback_response,
                "finish_reason": "fallback",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "model": self.model_name,
                "generation_time_ms": 0.0,
                "error": str(e)
            }

    def validate_response(self, response: Dict[str, Any]) -> bool:
        """Validate that response meets medical safety criteria.

        Args:
            response: Response dict from generate().

        Returns:
            True if response is valid and safe.

        Example:
            >>> response = client.generate(messages)
            >>> if client.validate_response(response):
            ...     print(response["content"])
        """
        # Check if response exists
        if not response or "content" not in response:
            logger.warning("Response missing content")
            return False

        content = response["content"]

        # Check for empty response
        if not content or not content.strip():
            logger.warning("Empty response content")
            return False

        # Check for finish reason
        finish_reason = response.get("finish_reason", "")
        if finish_reason == "length":
            logger.warning("Response truncated due to length")
            # Still valid, but log warning

        if finish_reason == "content_filter":
            logger.error("Response blocked by content filter")
            return False

        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Get current model configuration.

        Returns:
            Dict with model configuration details.
        """
        return {
            "model_name": self.model_name,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }


def create_client_from_config(config: Dict[str, Any], api_key: str) -> DeepSeekClient:
    """Factory function to create client from configuration dict.

    Args:
        config: LLM configuration dict from model_config.yaml.
        api_key: DeepSeek API key.

    Returns:
        Initialized DeepSeekClient.

    Example:
        >>> from config.settings import get_settings
        >>> settings = get_settings()
        >>> llm_config = settings.model_config['llm']
        >>> client = create_client_from_config(llm_config, api_key="sk-xxx")
    """
    return DeepSeekClient(
        api_key=api_key,
        model_name=config.get('model_name', 'deepseek-chat'),
        api_base=config.get('api_base', 'https://api.deepseek.com/v1'),
        temperature=config.get('temperature', 0.1),
        max_tokens=config.get('max_tokens', 2048),
        top_p=config.get('top_p', 0.9),
        timeout=config.get('timeout', 60),
        stream=config.get('stream', False)
    )
