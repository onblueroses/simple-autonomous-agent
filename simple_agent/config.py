"""Configuration types for the agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import openai

    from .quality import QualityRule


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
    scorer_prompt_template: str = (
        "Rate the relevance of this content on a scale of 0.0 to 1.0.\n"
        'Return ONLY a JSON object: {"score": <float>, "reason": "<brief>"}\n\n'
        "Content:\n{content}"
    )
    persona_select_prompt_template: str = (
        "Given this content, which persona should respond?\n"
        "Available: {personas}\n"
        'Return ONLY a JSON object: {"persona": "<name>", "reason": "<brief>"}\n\n'
        "Content:\n{content}"
    )


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
