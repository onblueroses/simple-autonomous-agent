"""LLM client with model routing.

Different models are good at different jobs. A 12B parameter model can score
relevance for pennies per thousand calls, but it can't write nuanced analysis.
A 235B thinking model produces great reasoning but is slow and expensive for
simple classification. A creative writing model generates fluent prose but
hallucinates if it doesn't have grounding context.

This module gives each job to the right model: score() for cheap classification,
reason() for deep analysis (with thinking-model support), draft() for persona-voiced
generation. All three use the same OpenAI-compatible client, so they work with
any provider: OpenRouter, Ollama, Together, vLLM.
"""

from __future__ import annotations

import asyncio
import re
import time

import openai

from .config import ModelConfig

# Exceptions worth retrying - transient failures that often resolve on their own
_RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
)


def _retry_llm_call(fn, *args, max_retries: int = 2, base_delay: float = 1.0, **kwargs):
    """Retry an LLM call on transient failures with exponential backoff.

    Only retries rate limits (429), timeouts, and connection errors. All other
    exceptions propagate immediately - retrying a malformed request or auth
    failure would just waste time.

    With max_retries=2 and base_delay=1.0: first retry after 1s, second after 2s.
    Total max wait: 3s for 3 attempts. Conservative enough for free-tier rate limits.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except _RETRYABLE_ERRORS as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** attempt))
            else:
                raise
    raise last_error  # type: ignore[misc]


def create_client(
    base_url: str,
    api_key: str,
    default_headers: dict[str, str] | None = None,
) -> openai.OpenAI:
    """Create an OpenAI-compatible client."""
    return openai.OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers=default_headers or {},
    )


def score(
    client: openai.OpenAI,
    prompt: str,
    config: ModelConfig,
    max_retries: int = 2,
    retry_base_delay: float = 1.0,
) -> str:
    """Single-turn completion for classification (scoring, persona selection)."""
    response = _retry_llm_call(
        client.chat.completions.create,
        model=config.model, max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[{"role": "user", "content": prompt}],
        max_retries=max_retries, base_delay=retry_base_delay,
    )
    return response.choices[0].message.content or ""


def reason(
    client: openai.OpenAI,
    prompt: str,
    config: ModelConfig,
    max_retries: int = 2,
    retry_base_delay: float = 1.0,
) -> str:
    """Single-turn completion with thinking-model support.

    If content is non-empty, strips <think> tags and returns. If empty, falls
    back to provider-specific fields: reasoning_content, then reasoning.
    """
    response = _retry_llm_call(
        client.chat.completions.create,
        model=config.model, max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[{"role": "user", "content": prompt}],
        max_retries=max_retries, base_delay=retry_base_delay,
    )
    message = response.choices[0].message
    content = message.content or ""

    # If content is non-empty, check for <think> tags and strip them
    if content:
        return _strip_think_tags(content)

    # Try provider-specific reasoning fields
    if hasattr(message, "reasoning_content") and message.reasoning_content:
        return message.reasoning_content
    if hasattr(message, "reasoning") and message.reasoning:
        return message.reasoning

    return content


# Regex to match <think>...</think> blocks (case-insensitive, dotall)
_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>(.*)", re.DOTALL | re.IGNORECASE)


def _strip_think_tags(content: str) -> str:
    """Strip <think> tags from model output.

    If there's content after the closing </think> tag, return that (the actual
    answer). Otherwise return the content inside the tags (some models put
    everything inside).
    """
    match = _THINK_TAG_RE.search(content)
    if not match:
        return content

    inside = match.group(1).strip()
    after = match.group(2).strip()
    return after if after else inside


def draft(
    client: openai.OpenAI,
    system_prompt: str,
    user_prompt: str,
    config: ModelConfig,
    max_retries: int = 2,
    retry_base_delay: float = 1.0,
) -> str:
    """System + user message completion for persona-voiced generation."""
    response = _retry_llm_call(
        client.chat.completions.create,
        model=config.model, max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_retries=max_retries, base_delay=retry_base_delay,
    )
    return response.choices[0].message.content or ""


# --- Async counterparts ---


async def _async_retry_llm_call(fn, *args, max_retries: int = 2, base_delay: float = 1.0, **kwargs):
    """Async version of _retry_llm_call. Uses await for the call and asyncio.sleep for backoff."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except _RETRYABLE_ERRORS as e:
            last_error = e
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** attempt))
            else:
                raise
    raise last_error  # type: ignore[misc]


def acreate_client(
    base_url: str,
    api_key: str,
    default_headers: dict[str, str] | None = None,
) -> openai.AsyncOpenAI:
    """Create an async OpenAI-compatible client. Constructor is sync."""
    return openai.AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers=default_headers or {},
    )


async def ascore(
    client: openai.AsyncOpenAI,
    prompt: str,
    config: ModelConfig,
    max_retries: int = 2,
    retry_base_delay: float = 1.0,
) -> str:
    """Async single-turn completion for classification."""
    response = await _async_retry_llm_call(
        client.chat.completions.create,
        model=config.model, max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[{"role": "user", "content": prompt}],
        max_retries=max_retries, base_delay=retry_base_delay,
    )
    return response.choices[0].message.content or ""


async def areason(
    client: openai.AsyncOpenAI,
    prompt: str,
    config: ModelConfig,
    max_retries: int = 2,
    retry_base_delay: float = 1.0,
) -> str:
    """Async single-turn completion with thinking-model support."""
    response = await _async_retry_llm_call(
        client.chat.completions.create,
        model=config.model, max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[{"role": "user", "content": prompt}],
        max_retries=max_retries, base_delay=retry_base_delay,
    )
    message = response.choices[0].message
    content = message.content or ""

    if content:
        return _strip_think_tags(content)

    if hasattr(message, "reasoning_content") and message.reasoning_content:
        return message.reasoning_content
    if hasattr(message, "reasoning") and message.reasoning:
        return message.reasoning

    return content


async def adraft(
    client: openai.AsyncOpenAI,
    system_prompt: str,
    user_prompt: str,
    config: ModelConfig,
    max_retries: int = 2,
    retry_base_delay: float = 1.0,
) -> str:
    """Async system + user message completion for persona-voiced generation."""
    response = await _async_retry_llm_call(
        client.chat.completions.create,
        model=config.model, max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_retries=max_retries, base_delay=retry_base_delay,
    )
    return response.choices[0].message.content or ""
