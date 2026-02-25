"""Core orchestration. Wires score-ground-reason-draft with per-stage fault tolerance."""

from __future__ import annotations

import json
import time

from . import llm
from .config import PipelineConfig, PipelineResult
from .persona import Persona, build_system_prompt
from .quality import default_rules, sanitize_input, validate_output
from .state import StateStore


def run_pipeline(
    item: dict,
    config: PipelineConfig,
    state: StateStore | None = None,
    personas: list[Persona] | None = None,
) -> PipelineResult:
    """Run the full pipeline for a single item. Each stage fails independently."""
    result = PipelineResult(item_id=item.get("id", "unknown"))
    errors: list[str] = []
    text = sanitize_input(item.get("text", ""))

    # Score
    try:
        score_prompt = (
            "Rate the relevance of this content on a scale of 0.0 to 1.0.\n"
            'Return ONLY a JSON object: {"score": <float>, "reason": "<brief>"}\n\n'
            f"Content:\n{text}"
        )
        parsed = json.loads(llm.score(config.scorer_client, score_prompt, config.scorer))
        result.score = float(parsed.get("score", 0.0))
    except Exception as e:
        errors.append(f"scoring: {e}")
        result.score = 0.0

    if result.score < config.score_threshold:
        result.errors = errors
        if state:
            state.save_item(result.item_id, item, result.score)
        return result

    # Select persona
    selected: Persona | None = None
    if personas and len(personas) > 1:
        try:
            names = ", ".join(p.name for p in personas)
            select_prompt = (
                f"Given this content, which persona should respond?\n"
                f"Available: {names}\n"
                'Return ONLY a JSON object: {"persona": "<name>", "reason": "<brief>"}\n\n'
                f"Content:\n{text}"
            )
            raw = llm.score(config.scorer_client, select_prompt, config.scorer)
            selected_name = json.loads(raw).get("persona", "").lower()
            selected = next((p for p in personas if p.name.lower() == selected_name), None)
        except Exception as e:
            errors.append(f"persona selection: {e}")

    if not selected and personas:
        selected = personas[0]
    if selected:
        result.persona = selected.name

    # Ground (optional)
    if config.ground_fn:
        try:
            result.grounding = config.ground_fn(text[:500])
        except Exception as e:
            errors.append(f"grounding: {e}")

    # Reason
    try:
        reason_parts = ["Analyze this content. Identify the core question, relevant facts, and what a response should address."]
        if result.grounding:
            reason_parts.append(f"\n<context>\n{result.grounding}\n</context>")
        reason_parts.append(f"\n<content>\n{text}\n</content>")
        result.reasoning = llm.reason(config.writer_client, "\n".join(reason_parts), config.reasoner)
    except Exception as e:
        errors.append(f"reasoning: {e}")

    # Draft
    try:
        system_prompt = build_system_prompt(selected) if selected else "You are a knowledgeable writer. Be direct and specific."
        user_parts = []
        if result.reasoning:
            user_parts.append(f"<analysis>\n{result.reasoning}\n</analysis>")
        if result.grounding:
            user_parts.append(f"<research>\n{result.grounding}\n</research>")
        user_parts.append(f"<content>\n{text}\n</content>")
        user_parts.append("\nWrite a response. Be direct and specific. Use concrete details from the analysis and research above.")
        result.draft = llm.draft(config.writer_client, system_prompt, "\n".join(user_parts), config.writer)
    except Exception as e:
        errors.append(f"drafting: {e}")

    # Validate
    rules = config.quality_rules or default_rules()
    passed, reasons = validate_output(result.draft, rules)
    result.passed_quality = passed
    errors.extend(reasons)

    # Persist
    if state:
        try:
            state.save_item(result.item_id, item, result.score)
            if result.draft:
                state.save_draft(result.item_id, result.persona or "default", result.draft)
                state.update_item_status(result.item_id, "drafted")
        except Exception as e:
            errors.append(f"persistence: {e}")

    result.errors = errors
    return result


def run_batch(
    items: list[dict],
    config: PipelineConfig,
    state: StateStore | None = None,
    personas: list[Persona] | None = None,
    delay: float = 1.0,
) -> list[PipelineResult]:
    """Run the pipeline for a batch of items with rate limiting between calls."""
    results: list[PipelineResult] = []
    run_id = state.start_run() if state else None
    drafts_created = 0

    for i, item in enumerate(items):
        result = run_pipeline(item, config, state, personas)
        results.append(result)
        if result.draft:
            drafts_created += 1
        if i < len(items) - 1 and delay > 0:
            time.sleep(delay)

    if state and run_id is not None:
        all_errors = [e for r in results for e in r.errors]
        state.finish_run(run_id, len(items), drafts_created, all_errors)

    return results
