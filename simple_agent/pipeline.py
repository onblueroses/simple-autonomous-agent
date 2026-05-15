"""Core orchestration. Wires score-ground-reason-draft with per-stage fault tolerance.

Score-first filtering saves ~50x on items that don't pass the threshold (small
model classification vs full reasoning + drafting). XML tags in prompts delimit
trusted context from untrusted user content to reduce prompt injection surface.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
from typing import TypedDict

from . import llm
from .config import AsyncPipelineConfig, PipelineConfig, PipelineResult
from .persona import Persona, build_system_prompt
from .quality import default_rules, sanitize_input, validate_output
from .state import StateStore


class _RetryKwargs(TypedDict):
    max_retries: int
    retry_base_delay: float


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_SMART_QUOTES = str.maketrans(
    {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"}
)
_LINE_COMMENT_RE = re.compile(r"(?m)^\s*//[^\n]*")
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _sanitize_json_text(s: str) -> str:
    """Strip common LLM JSON defects: smart quotes, full-line // comments, trailing commas.

    The line-comment stripper only removes lines that begin with `//` (after optional
    whitespace). It deliberately does NOT touch `//` mid-line, since URLs and natural-
    language strings containing `//` would be corrupted.
    """
    s = s.translate(_SMART_QUOTES)
    s = _LINE_COMMENT_RE.sub("", s)
    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    return s


def _extract_json(text: str) -> str:
    text = text.strip()

    fence_match = _FENCED_JSON_RE.search(text)
    if fence_match:
        return _sanitize_json_text(fence_match.group(1).strip())

    if text.startswith("{") or text.startswith("["):
        return _sanitize_json_text(text)

    brace_idx = text.find("{")
    if brace_idx != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(brace_idx, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return _sanitize_json_text(text[brace_idx : i + 1])

    return text


def _parse_score(raw: str) -> float:
    return float(json.loads(_extract_json(raw)).get("score", 0.0))


def _resolve_persona(raw: str, personas: list[Persona]) -> Persona | None:
    name = json.loads(_extract_json(raw)).get("persona", "").lower()
    return next((p for p in personas if p.name.lower() == name), None)


def _build_reason_prompt(text: str, grounding: str) -> str:
    parts = [
        "Analyze this content. Identify the core question, relevant facts, and what a response should address."
    ]
    if grounding:
        parts.append(f"\n<context>\n{grounding}\n</context>")
    parts.append(f"\n<content>\n{text}\n</content>")
    return "\n".join(parts)


def _build_draft_prompt(reasoning: str, grounding: str, text: str) -> str:
    parts = []
    if reasoning:
        parts.append(f"<analysis>\n{reasoning}\n</analysis>")
    if grounding:
        parts.append(f"<research>\n{grounding}\n</research>")
    parts.append(f"<content>\n{text}\n</content>")
    parts.append(
        "\nWrite a response. Be direct and specific. Use concrete details from the analysis and research above."
    )
    return "\n".join(parts)


def _validate_and_persist(
    result: PipelineResult,
    item: dict,
    config: PipelineConfig | AsyncPipelineConfig,
    state: StateStore | None,
    errors: list[str],
    persona: Persona | None = None,
) -> PipelineResult:
    # Per-persona rules OVERRIDE the pipeline default — they replace, not merge.
    if persona is not None and persona.quality_rules is not None:
        rules = persona.quality_rules
    else:
        rules = config.quality_rules or default_rules()
    reasons = validate_output(result.draft, rules)
    result.passed_quality = len(reasons) == 0
    errors.extend(reasons)

    if state:
        try:
            state.save_item(result.item_id, item, result.score)
            if result.draft:
                state.save_draft(
                    result.item_id, result.persona or "default", result.draft
                )
                state.update_item_status(result.item_id, "drafted")
        except Exception as e:
            errors.append(f"persistence: {e}")

    result.errors = errors
    return result


def _init_pipeline(
    item: dict,
    config: PipelineConfig | AsyncPipelineConfig,
) -> tuple[PipelineResult, list[str], str]:
    return (
        PipelineResult(item_id=item.get("id", "unknown")),
        [],
        sanitize_input(item.get("text", "")),
    )


def _apply_persona(
    selected: Persona | None, personas: list[Persona] | None, result: PipelineResult
) -> Persona | None:
    """Fall back to first persona if selection failed, then record on result."""
    if not selected and personas:
        selected = personas[0]
    if selected:
        result.persona = selected.name
    return selected


def _should_skip_search(
    config: PipelineConfig,
    item: dict,
    selected: Persona | None,
    errors: list[str],
) -> bool:
    if config.needs_search is None:
        return False
    try:
        return not bool(config.needs_search(item, selected, config))
    except Exception as e:
        errors.append(f"needs_search: {e}")
        return False


async def _async_should_skip_search(
    config: AsyncPipelineConfig,
    item: dict,
    selected: Persona | None,
    errors: list[str],
) -> bool:
    if config.needs_search is None:
        return False
    try:
        raw = config.needs_search(item, selected, config)
        if inspect.isawaitable(raw):
            raw = await raw
        return not bool(raw)
    except Exception as e:
        errors.append(f"needs_search: {e}")
        return False


def _finish_below_threshold(
    result: PipelineResult,
    item: dict,
    errors: list[str],
    state: StateStore | None,
) -> PipelineResult:
    result.errors = errors
    if state:
        state.save_item(result.item_id, item, result.score)
    return result


def run_pipeline(
    item: dict,
    config: PipelineConfig,
    state: StateStore | None = None,
    personas: list[Persona] | None = None,
) -> PipelineResult:
    result, errors, text = _init_pipeline(item, config)
    retry: _RetryKwargs = {
        "max_retries": config.max_retries,
        "retry_base_delay": config.retry_base_delay,
    }

    # Score
    try:
        score_prompt = config.scorer_prompt_template.replace("{content}", text)
        result.score = _parse_score(
            llm.score(config.scorer_client, score_prompt, config.scorer, **retry)
        )
    except Exception as e:
        errors.append(f"scoring: {e}")
        result.score = 0.0

    if result.score < config.score_threshold:
        return _finish_below_threshold(result, item, errors, state)

    # Select persona
    selected: Persona | None = None
    if personas and len(personas) > 1:
        try:
            names = ", ".join(p.name for p in personas)
            select_prompt = config.persona_select_prompt_template.replace(
                "{personas}", names
            ).replace("{content}", text)
            selected = _resolve_persona(
                llm.score(config.scorer_client, select_prompt, config.scorer, **retry),
                personas,
            )
        except Exception as e:
            errors.append(f"persona selection: {e}")
    selected = _apply_persona(selected, personas, result)

    # Ground (gated by optional needs_search)
    if config.ground_fn and not _should_skip_search(config, item, selected, errors):
        try:
            result.grounding = config.ground_fn(text[:500])
        except Exception as e:
            errors.append(f"grounding: {e}")

    # Reason
    try:
        result.reasoning = llm.reason(
            config.writer_client,
            _build_reason_prompt(text, result.grounding),
            config.reasoner,
            **retry,
        )
    except Exception as e:
        errors.append(f"reasoning: {e}")

    # Draft
    try:
        system_prompt = (
            build_system_prompt(selected)
            if selected
            else "You are a knowledgeable writer. Be direct and specific."
        )
        result.draft = llm.draft(
            config.writer_client,
            system_prompt,
            _build_draft_prompt(result.reasoning, result.grounding, text),
            config.writer,
            **retry,
        )
    except Exception as e:
        errors.append(f"drafting: {e}")

    return _validate_and_persist(result, item, config, state, errors, persona=selected)


def _finalize_batch(
    state: StateStore | None,
    run_id: int | None,
    results: list[PipelineResult],
    n_items: int,
) -> None:
    if state and run_id is not None:
        drafts_created = sum(1 for r in results if r.draft)
        all_errors = [e for r in results for e in r.errors]
        state.finish_run(run_id, n_items, drafts_created, all_errors)


def _batch_iter_sync(
    items: list[dict],
    config: PipelineConfig,
    state: StateStore | None,
    personas: list[Persona] | None,
    delay: float,
) -> list[PipelineResult]:
    results: list[PipelineResult] = []
    for i, item in enumerate(items):
        results.append(run_pipeline(item, config, state, personas))
        if i < len(items) - 1 and delay > 0:
            time.sleep(delay)
    return results


async def _batch_iter_async(
    items: list[dict],
    config: AsyncPipelineConfig,
    state: StateStore | None,
    personas: list[Persona] | None,
    max_concurrency: int,
) -> list[PipelineResult]:
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(item: dict) -> PipelineResult:
        async with sem:
            return await arun_pipeline(item, config, state, personas)

    return list(await asyncio.gather(*[_one(item) for item in items]))


def run_batch(
    items: list[dict],
    config: PipelineConfig,
    state: StateStore | None = None,
    personas: list[Persona] | None = None,
    delay: float = 1.0,
) -> list[PipelineResult]:
    run_id = state.start_run() if state else None
    results = _batch_iter_sync(items, config, state, personas, delay)
    _finalize_batch(state, run_id, results, len(items))
    return results


async def arun_pipeline(
    item: dict,
    config: AsyncPipelineConfig,
    state: StateStore | None = None,
    personas: list[Persona] | None = None,
) -> PipelineResult:
    result, errors, text = _init_pipeline(item, config)
    retry: _RetryKwargs = {
        "max_retries": config.max_retries,
        "retry_base_delay": config.retry_base_delay,
    }

    # Score
    try:
        score_prompt = config.scorer_prompt_template.replace("{content}", text)
        result.score = _parse_score(
            await llm.ascore(config.scorer_client, score_prompt, config.scorer, **retry)
        )
    except Exception as e:
        errors.append(f"scoring: {e}")
        result.score = 0.0

    if result.score < config.score_threshold:
        return _finish_below_threshold(result, item, errors, state)

    # Select persona
    selected: Persona | None = None
    if personas and len(personas) > 1:
        try:
            names = ", ".join(p.name for p in personas)
            select_prompt = config.persona_select_prompt_template.replace(
                "{personas}", names
            ).replace("{content}", text)
            selected = _resolve_persona(
                await llm.ascore(
                    config.scorer_client, select_prompt, config.scorer, **retry
                ),
                personas,
            )
        except Exception as e:
            errors.append(f"persona selection: {e}")
    selected = _apply_persona(selected, personas, result)

    # Ground (gated by optional needs_search; supports sync or async predicate)
    if config.ground_fn and not await _async_should_skip_search(
        config, item, selected, errors
    ):
        try:
            result.grounding = await config.ground_fn(text[:500])
        except Exception as e:
            errors.append(f"grounding: {e}")

    # Reason
    try:
        result.reasoning = await llm.areason(
            config.writer_client,
            _build_reason_prompt(text, result.grounding),
            config.reasoner,
            **retry,
        )
    except Exception as e:
        errors.append(f"reasoning: {e}")

    # Draft
    try:
        system_prompt = (
            build_system_prompt(selected)
            if selected
            else "You are a knowledgeable writer. Be direct and specific."
        )
        result.draft = await llm.adraft(
            config.writer_client,
            system_prompt,
            _build_draft_prompt(result.reasoning, result.grounding, text),
            config.writer,
            **retry,
        )
    except Exception as e:
        errors.append(f"drafting: {e}")

    return _validate_and_persist(result, item, config, state, errors, persona=selected)


async def arun_batch(
    items: list[dict],
    config: AsyncPipelineConfig,
    state: StateStore | None = None,
    personas: list[Persona] | None = None,
    max_concurrency: int = 5,
) -> list[PipelineResult]:
    run_id = state.start_run() if state else None
    results = await _batch_iter_async(items, config, state, personas, max_concurrency)
    _finalize_batch(state, run_id, results, len(items))
    return results
