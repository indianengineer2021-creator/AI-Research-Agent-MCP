"""Groq chat model construction and resilient invocation helpers."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import TypeVar

from langchain_groq import ChatGroq

from config import DEFAULT_MODEL

logger = logging.getLogger(__name__)
T = TypeVar("T")

TRANSIENT_ERROR_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "RateLimitError",
    "ServiceUnavailableError",
}


class MissingAPIKeyError(ValueError):
    """Raised if a Groq API key was not configured."""


class LLMTokenLimitError(RuntimeError):
    """Raised when Groq rejects a request for exceeding its token quota."""


def create_llm(
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> ChatGroq:
    """Create a Groq chat model, preferring an explicit UI key over the env key."""
    resolved_key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()
    if not resolved_key:
        raise MissingAPIKeyError(
            "A Groq API key is required. Enter one in the sidebar or set "
            "GROQ_API_KEY in your .env file."
        )
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("Groq temperature must be between 0.0 and 2.0.")

    logger.info("Creating ChatGroq model %s", model)
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=resolved_key,
        max_retries=0,
    )


def is_transient_llm_error(error: Exception) -> bool:
    """Return whether an exception is a rate-limit or temporary API failure."""
    if type(error).__name__ in TRANSIENT_ERROR_NAMES:
        return True
    return "429" in str(error) or "rate limit" in str(error).casefold()


async def invoke_with_retry(
    invoke: Callable[..., Awaitable[T]],
    value: object,
    *,
    operation: str,
    attempts: int = 3,
) -> T:
    """Retry temporary model failures with capped exponential backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return await invoke(value)
        except Exception as exc:
            if getattr(exc, "status_code", None) == 413 or (
                "tokens per minute" in str(exc).casefold()
            ):
                logger.warning("Groq token limit reached during '%s'", operation)
                raise LLMTokenLimitError(
                    "Groq rejected this request because the account's token limit "
                    "was reached. Wait about a minute and retry, or choose a smaller "
                    "model and reduce the maximum search iterations."
                ) from exc
            if not is_transient_llm_error(exc) or attempt == attempts:
                logger.exception(
                    "LLM operation '%s' failed on attempt %d/%d",
                    operation,
                    attempt,
                    attempts,
                )
                raise

            delay = min(2 ** (attempt - 1), 8)
            logger.warning(
                "Temporary LLM failure during '%s' (attempt %d/%d); retrying in %d s",
                operation,
                attempt,
                attempts,
                delay,
            )
            await asyncio.sleep(delay)

    raise RuntimeError(f"LLM operation '{operation}' exhausted its retries.")
