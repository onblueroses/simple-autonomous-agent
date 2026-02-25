"""Core orchestration. Wires score-ground-reason-draft with per-stage fault tolerance.

Score-first filtering saves ~50x on items that don't pass the threshold (small
model classification vs full reasoning + drafting). XML tags in prompts delimit
trusted context from untrusted user content to reduce prompt injection surface.
"""

from __future__ import annotations

import json
import re
import time

from . import llm
from .config import PipelineConfig, PipelineResult
from .persona import Persona, build_system_prompt
from .quality import default_rules, sanitize_input, validate_output
from .state import StateStore

# Regex to match JSON wrapped in triple-backtick fences (```json ... ``` or ``` ... ```)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _extract_json(text: str) -> str:
    """Extract JSON from LLM output that may be wrapped in markdown fences or preamble.

    LLMs frequently return JSON in three ways:
    1. Clean JSON: {"score": 0.8} - pass through unchanged
    2. Fenced: ```json\n{"score": 0.8}\n``` - strip the fences
    3. With preamble: "Here's the result:\n{"score": 0.8}" - find the first { and match to }

    Returns the extracted string for json.loads(). If no JSON structure is detected,
    returns the original text (which will fail at json.loads as before).
    """
    text = text.strip()

    # Try fenced extraction first
    fence_match = _FENCED_JSON_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()

    # If text already starts with { or [, it's probably clean JSON
    if text.startswith("{") or text.startswith("["):
        return text

    # Try to find the first JSON object in preamble text
    brace_idx = text.find("{")
    if brace_idx != -1:
        # Find the matching closing brace by counting nesting
        depth = 0
        for i in range(brace_idx, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[brace_idx : i + 1]

    # Nothing found - return original (will fail at json.loads, caught by try/except)
    return text


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
        score_prompt = config.scorer_prompt_template.replace("{content}", text)
        raw_score = llm.score(
            config.scorer_client, score_prompt, config.scorer,
            max_retries=config.max_retries, retry_base_delay=config.retry_base_delay,
        )
        parsed = json.loads(_extract_json(raw_score))
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
            select_prompt = config.persona_select_prompt_template.replace(
                "{personas}", names,
            ).replace("{content}", text)
            raw = llm.score(
                config.scorer_client, select_prompt, config.scorer,
                max_retries=config.max_retries, retry_base_delay=config.retry_base_delay,
            )
            selected_name = json.loads(_extract_json(raw)).get("persona", "").lower()
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
        result.reasoning = llm.reason(
            config.writer_client, "\n".join(reason_parts), config.reasoner,
            max_retries=config.max_retries, retry_base_delay=config.retry_base_delay,
        )
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
        result.draft = llm.draft(
            config.writer_client, system_prompt, "\n".join(user_parts), config.writer,
            max_retries=config.max_retries, retry_base_delay=config.retry_base_delay,
        )
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
