"""Core pipeline orchestration.

The centerpiece: wires score-ground-reason-draft into a single function
with fault tolerance at every stage. A failed search doesn't kill the draft.
A failed reasoning step just means the writer works without analysis.

This is where multi-model orchestration happens - each stage uses a
different model chosen for what it's best at.
"""

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
    """Run the full score-ground-reason-draft pipeline for a single item.

    Each stage is wrapped in try/except. Failures degrade gracefully -
    a failed grounding step doesn't prevent drafting, it just means
    the draft won't have external context.

    Args:
        item: Dict with at least 'id' and 'text' keys.
        config: Pipeline configuration with model configs and clients.
        state: Optional StateStore for persistence.
        personas: Optional list of personas. First is used if only one.

    Returns:
        PipelineResult with all intermediate outputs and any errors.
    """
    result = PipelineResult(item_id=item.get("id", "unknown"))
    errors: list[str] = []

    # --- Stage 1: Score ---
    try:
        score_prompt = (
            "Rate the relevance of this content on a scale of 0.0 to 1.0.\n"
            "Return ONLY a JSON object: {\"score\": <float>, \"reason\": \"<brief>\"}\n\n"
            f"Content:\n{sanitize_input(item.get('text', ''))}"
        )
        raw_score = llm.score(config.scorer_client, score_prompt, config.scorer)
        parsed = json.loads(raw_score)
        result.score = float(parsed.get("score", 0.0))
    except Exception as e:
        errors.append(f"scoring: {e}")
        result.score = 0.0

    if result.score < config.score_threshold:
        result.passed_quality = False
        result.errors = errors
        if state:
            state.save_item(result.item_id, item, result.score)
        return result

    # --- Stage 2: Select persona ---
    selected_persona: Persona | None = None
    if personas and len(personas) > 1:
        try:
            persona_names = ", ".join(p.name for p in personas)
            select_prompt = (
                f"Given this content, which persona should respond?\n"
                f"Available: {persona_names}\n"
                f"Return ONLY a JSON object: {{\"persona\": \"<name>\", \"reason\": \"<brief>\"}}\n\n"
                f"Content:\n{sanitize_input(item.get('text', ''))}"
            )
            raw_selection = llm.score(config.scorer_client, select_prompt, config.scorer)
            parsed_selection = json.loads(raw_selection)
            selected_name = parsed_selection.get("persona", "")
            for p in personas:
                if p.name.lower() == selected_name.lower():
                    selected_persona = p
                    break
        except Exception as e:
            errors.append(f"persona selection: {e}")

    if not selected_persona and personas:
        selected_persona = personas[0]

    if selected_persona:
        result.persona = selected_persona.name

    # --- Stage 3: Ground (optional) ---
    if config.ground_fn:
        try:
            query = item.get("text", "")[:500]
            result.grounding = config.ground_fn(query)
        except Exception as e:
            errors.append(f"grounding: {e}")
            result.grounding = ""

    # --- Stage 4: Reason ---
    try:
        reason_parts = ["Analyze the following content. Identify the core question, relevant facts, and what a response should address."]
        if result.grounding:
            reason_parts.append(f"\n<context>\n{result.grounding}\n</context>")
        reason_parts.append(f"\n<content>\n{sanitize_input(item.get('text', ''))}\n</content>")

        result.reasoning = llm.reason(
            config.writer_client,
            "\n".join(reason_parts),
            config.reasoner,
        )
    except Exception as e:
        errors.append(f"reasoning: {e}")
        result.reasoning = ""

    # --- Stage 5: Draft ---
    try:
        system_prompt = ""
        if selected_persona:
            system_prompt = build_system_prompt(selected_persona)
        else:
            system_prompt = "You are a knowledgeable writer. Be direct and specific."

        user_parts = []
        if result.reasoning:
            user_parts.append(f"<analysis>\n{result.reasoning}\n</analysis>")
        if result.grounding:
            user_parts.append(f"<research>\n{result.grounding}\n</research>")
        user_parts.append(f"<content>\n{sanitize_input(item.get('text', ''))}\n</content>")
        user_parts.append("\nWrite a response. Be direct and specific. Use concrete details from the analysis and research above.")

        result.draft = llm.draft(
            config.writer_client,
            system_prompt,
            "\n".join(user_parts),
            config.writer,
        )
    except Exception as e:
        errors.append(f"drafting: {e}")
        result.draft = ""

    # --- Stage 6: Validate ---
    rules = config.quality_rules or default_rules()
    passed, reasons = validate_output(result.draft, rules)
    result.passed_quality = passed
    if reasons:
        errors.extend(reasons)

    # --- Stage 7: Persist (optional) ---
    if state:
        try:
            state.save_item(result.item_id, item, result.score)
            if result.draft:
                state.save_draft(
                    result.item_id,
                    result.persona or "default",
                    result.draft,
                )
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
    """Run the pipeline for a batch of items.

    Processes all items even if individual ones fail. Optionally logs
    a run to the state store for observability.
    """
    results: list[PipelineResult] = []
    run_id = state.start_run() if state else None

    drafts_created = 0
    all_errors: list[str] = []

    for i, item in enumerate(items):
        result = run_pipeline(item, config, state, personas)
        results.append(result)

        if result.draft:
            drafts_created += 1
        all_errors.extend(result.errors)

        # Rate limiting pause between items
        if i < len(items) - 1 and delay > 0:
            time.sleep(delay)

    if state and run_id is not None:
        state.finish_run(run_id, len(items), drafts_created, all_errors)

    return results
