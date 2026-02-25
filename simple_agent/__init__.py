"""simple-autonomous-agent: Multi-model LLM orchestration in ~500 lines.

A small library demonstrating how production autonomous agents actually work.
Not a framework - a pattern you can read in 20 minutes.
"""

from .config import ModelConfig, PipelineConfig, PipelineResult
from .llm import create_client, draft, reason, score
from .persona import Persona, build_system_prompt, list_personas, load_persona
from .pipeline import run_batch, run_pipeline
from .quality import (
    QualityRule,
    Violation,
    check_quality,
    default_rules,
    sanitize_input,
    validate_output,
)
from .state import StateStore

__all__ = [
    # Config
    "ModelConfig",
    "PipelineConfig",
    "PipelineResult",
    # LLM
    "create_client",
    "score",
    "reason",
    "draft",
    # Persona
    "Persona",
    "load_persona",
    "build_system_prompt",
    "list_personas",
    # Pipeline
    "run_pipeline",
    "run_batch",
    # Quality
    "QualityRule",
    "Violation",
    "check_quality",
    "default_rules",
    "sanitize_input",
    "validate_output",
    # State
    "StateStore",
]
