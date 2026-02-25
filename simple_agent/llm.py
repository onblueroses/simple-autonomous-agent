"""LLM client with model routing.

Works with any OpenAI-compatible provider: OpenRouter, Ollama, Together, vLLM.
"""

from __future__ import annotations

import openai

from .config import ModelConfig


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


def score(client: openai.OpenAI, prompt: str, config: ModelConfig) -> str:
    """Single-turn completion. Returns raw response text."""
    response = client.chat.completions.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def reason(client: openai.OpenAI, prompt: str, config: ModelConfig) -> str:
    """Single-turn completion with thinking-model fallback.

    Some models (DeepSeek R1, Qwen3) return empty content with output
    in a separate `reasoning` field. This handles both.
    """
    response = client.chat.completions.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    message = response.choices[0].message
    content = message.content or ""
    if not content and hasattr(message, "reasoning") and message.reasoning:
        content = message.reasoning
    return content


def draft(
    client: openai.OpenAI,
    system_prompt: str,
    user_prompt: str,
    config: ModelConfig,
) -> str:
    """System + user message completion. System carries persona identity."""
    response = client.chat.completions.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""
