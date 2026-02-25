"""Configuration types for the agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from collections.abc import Awaitable

    import openai

    from .quality import QualityRule

_DEFAULT_SCORER_PROMPT = (
    "Rate the relevance of this content on a scale of 0.0 to 1.0.\n"
    'Return ONLY a JSON object: {"score": <float>, "reason": "<brief>"}\n\n'
    "Content:\n{content}"
)
_DEFAULT_PERSONA_SELECT_PROMPT = (
    "Given this content, which persona should respond?\n"
    "Available: {personas}\n"
    'Return ONLY a JSON object: {"persona": "<name>", "reason": "<brief>"}\n\n'
    "Content:\n{content}"
)


@dataclass
class ModelConfig:
    """Configuration for a single LLM model."""

    model: str
    max_tokens: int = 1024
    temperature: float | None = None


@dataclass
class PipelineConfig:
    """Configuration for the full score-ground-reason-draft pipeline."""

    scorer: ModelConfig
    reasoner: ModelConfig
    writer: ModelConfig
    scorer_client: openai.OpenAI
    writer_client: openai.OpenAI
    quality_rules: list[QualityRule] = field(default_factory=list)
    ground_fn: Callable[[str], str] | None = None
    score_threshold: float = 0.6
    max_retries: int = 2
    retry_base_delay: float = 1.0
    scorer_prompt_template: str = _DEFAULT_SCORER_PROMPT
    persona_select_prompt_template: str = _DEFAULT_PERSONA_SELECT_PROMPT


@dataclass
class AsyncPipelineConfig:
    """Configuration for the async score-ground-reason-draft pipeline.

    Separate from PipelineConfig so the type checker catches a sync client
    passed to an async pipeline at definition time, not runtime.
    """

    scorer: ModelConfig
    reasoner: ModelConfig
    writer: ModelConfig
    scorer_client: openai.AsyncOpenAI
    writer_client: openai.AsyncOpenAI
    quality_rules: list[QualityRule] = field(default_factory=list)
    ground_fn: Callable[[str], Awaitable[str]] | None = None
    score_threshold: float = 0.6
    max_retries: int = 2
    retry_base_delay: float = 1.0
    scorer_prompt_template: str = _DEFAULT_SCORER_PROMPT
    persona_select_prompt_template: str = _DEFAULT_PERSONA_SELECT_PROMPT


@dataclass
class PipelineResult:
    """Result from a single pipeline run."""

    item_id: str
    score: float = 0.0
    grounding: str = ""
    reasoning: str = ""
    draft: str = ""
    persona: str = ""
    passed_quality: bool = False
    errors: list[str] = field(default_factory=list)
