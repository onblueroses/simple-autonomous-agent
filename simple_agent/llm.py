"""LLM client with model routing.

Three specialized functions for three jobs: scoring, reasoning, and drafting.
Each wraps the OpenAI-compatible API with the right call pattern for its task.
Works with any provider that speaks the OpenAI chat completions format -
OpenRouter, Ollama, Together, vLLM, or OpenAI itself.
"""

from __future__ import annotations

import openai

from .config import ModelConfig


def create_client(
    base_url: str,
    api_key: str,
    default_headers: dict[str, str] | None = None,
) -> openai.OpenAI:
    """Create an OpenAI-compatible client for any provider."""
    return openai.OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers=default_headers or {},
    )


def score(client: openai.OpenAI, prompt: str, config: ModelConfig) -> str:
    """Score content relevance. Uses a cheap, fast model.

    Returns the raw model response text. Caller parses the score.
    """
    response = client.chat.completions.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def reason(client: openai.OpenAI, prompt: str, config: ModelConfig) -> str:
    """Analyze content with a thinking model.

    Thinking models may put all output in a reasoning field instead of
    content. This function handles both patterns transparently.
    """
    response = client.chat.completions.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    message = response.choices[0].message
    # Thinking models sometimes return empty content with reasoning
    # in a separate field (e.g., DeepSeek R1, Qwen3).
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
    """Generate output with persona identity in the system message.

    The system prompt carries persona identity ("you ARE this person").
    The user prompt carries the task and any grounding/reasoning context.
    This separation is what makes persona voice consistent.
    """
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
