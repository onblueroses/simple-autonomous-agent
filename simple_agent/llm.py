"""LLM client with model routing.

score() for cheap classification, reason() for deep analysis (thinking-model
support), draft() for persona-voiced generation. All three use the OpenAI-
compatible client interface, so they work with any provider.

Sync/async pairs share message construction, request shape, retry-decision
logic, and response post-processing through module-level helpers. The pair's
body is only the call boundary that decides between sync and `await`.
"""

from __future__ import annotations

import asyncio
import re
import time

import openai

from .config import ModelConfig

_RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
)


def _user_messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def _system_user_messages(system_prompt: str, user_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _chat_kwargs(messages: list[dict], config: ModelConfig) -> dict:
    return {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "messages": messages,
    }


def _content_or_empty(response) -> str:
    return response.choices[0].message.content or ""


def _backoff_seconds(attempt: int, base_delay: float) -> float:
    return base_delay * (2**attempt)


def _retry_llm_call(fn, *args, max_retries: int = 2, base_delay: float = 1.0, **kwargs):
    last_error: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except _RETRYABLE_ERRORS as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(_backoff_seconds(attempt, base_delay))
            else:
                raise
    raise last_error  # type: ignore[misc]


async def _async_retry_llm_call(
    fn, *args, max_retries: int = 2, base_delay: float = 1.0, **kwargs
):
    last_error: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except _RETRYABLE_ERRORS as e:
            last_error = e
            if attempt < max_retries:
                await asyncio.sleep(_backoff_seconds(attempt, base_delay))
            else:
                raise
    raise last_error  # type: ignore[misc]


_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>(.*)", re.DOTALL | re.IGNORECASE)


def _extract_reasoning(message) -> str:
    content = message.content or ""
    if content:
        return _strip_think_tags(content)
    if hasattr(message, "reasoning_content") and message.reasoning_content:
        return message.reasoning_content
    if hasattr(message, "reasoning") and message.reasoning:
        return message.reasoning
    return content


def _strip_think_tags(content: str) -> str:
    match = _THINK_TAG_RE.search(content)
    if not match:
        return content
    inside = match.group(1).strip()
    after = match.group(2).strip()
    return after if after else inside


def create_client(
    base_url: str,
    api_key: str,
    default_headers: dict[str, str] | None = None,
) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers=default_headers or {},
    )


def acreate_client(
    base_url: str,
    api_key: str,
    default_headers: dict[str, str] | None = None,
) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
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
    resp = _retry_llm_call(
        client.chat.completions.create,
        max_retries=max_retries,
        base_delay=retry_base_delay,
        **_chat_kwargs(_user_messages(prompt), config),
    )
    return _content_or_empty(resp)


async def ascore(
    client: openai.AsyncOpenAI,
    prompt: str,
    config: ModelConfig,
    max_retries: int = 2,
    retry_base_delay: float = 1.0,
) -> str:
    resp = await _async_retry_llm_call(
        client.chat.completions.create,
        max_retries=max_retries,
        base_delay=retry_base_delay,
        **_chat_kwargs(_user_messages(prompt), config),
    )
    return _content_or_empty(resp)


def reason(
    client: openai.OpenAI,
    prompt: str,
    config: ModelConfig,
    max_retries: int = 2,
    retry_base_delay: float = 1.0,
) -> str:
    resp = _retry_llm_call(
        client.chat.completions.create,
        max_retries=max_retries,
        base_delay=retry_base_delay,
        **_chat_kwargs(_user_messages(prompt), config),
    )
    return _extract_reasoning(resp.choices[0].message)


async def areason(
    client: openai.AsyncOpenAI,
    prompt: str,
    config: ModelConfig,
    max_retries: int = 2,
    retry_base_delay: float = 1.0,
) -> str:
    resp = await _async_retry_llm_call(
        client.chat.completions.create,
        max_retries=max_retries,
        base_delay=retry_base_delay,
        **_chat_kwargs(_user_messages(prompt), config),
    )
    return _extract_reasoning(resp.choices[0].message)


def draft(
    client: openai.OpenAI,
    system_prompt: str,
    user_prompt: str,
    config: ModelConfig,
    max_retries: int = 2,
    retry_base_delay: float = 1.0,
) -> str:
    resp = _retry_llm_call(
        client.chat.completions.create,
        max_retries=max_retries,
        base_delay=retry_base_delay,
        **_chat_kwargs(_system_user_messages(system_prompt, user_prompt), config),
    )
    return _content_or_empty(resp)


async def adraft(
    client: openai.AsyncOpenAI,
    system_prompt: str,
    user_prompt: str,
    config: ModelConfig,
    max_retries: int = 2,
    retry_base_delay: float = 1.0,
) -> str:
    resp = await _async_retry_llm_call(
        client.chat.completions.create,
        max_retries=max_retries,
        base_delay=retry_base_delay,
        **_chat_kwargs(_system_user_messages(system_prompt, user_prompt), config),
    )
    return _content_or_empty(resp)
