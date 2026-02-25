"""Configuration types for the agent pipeline.

All config is explicit dataclasses - no env var magic, no global state.
Users instantiate these directly and pass them to pipeline functions.
"""

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
