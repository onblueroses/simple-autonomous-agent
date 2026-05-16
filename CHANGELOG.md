# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-05-16

### Fixed
- `tool_spec()` no longer raises `AttributeError` when given a bare
  `functools.partial` tool. The tool name is derived from the underlying
  function via `_unwrap_for_hints`. Documented async-section claim about
  partial-wrapped tools now matches the actual behavior.

## [0.3.0] - 2026-05-16

### Added
- `simple_agent.Agent` and `simple_agent.AsyncAgent` — minimal tool-calling loops
  wrapping any OpenAI-compatible client. Tools are plain callables; the OpenAI
  tool spec is auto-derived from type hints (`str`, `int`, `float`, `bool`,
  `list[T]`, `Optional[T]`, `T | None`). Memory is caller-owned and mutated in
  place. Termination reasons: `"done"`, `"max_steps"`, `"no_progress"`. Tool
  errors become observations so the loop can recover. Sync `Agent` rejects
  awaitable tool results; `AsyncAgent` awaits them.
- `simple_agent.AgentResult` — dataclass with `output`, `messages`, `steps`,
  `terminated`.
- `simple_agent.tool_spec(fn)` — public helper that returns the OpenAI tool
  spec for a single callable. Useful for introspection and composition.
- `examples/agent_real_api.py` — Agent loop with a sandboxed arithmetic
  evaluator (AST walker, no `eval`/`exec`) and a constant lookup table.
- `tests/test_agent.py` and `tests/test_agent_async.py` — 43 new tests covering
  schema derivation, tool dispatch, memory, retries, and termination.

### Changed
- `README.md` repositioned around the Agent class. The content pipeline is now
  documented under "Specialized: Content Pipeline"; all v0.2.0 pipeline
  content is preserved.
- `scripts/check_dedup.py` PAIRS list extended with the new `run`/`arun` pair;
  dedup gate now enforces parity across 7 sync/async pairs.
- `tests/test_api_contract.py` `_ALLOWED_NEW_EXPORTS` widened to admit
  `Agent`, `AsyncAgent`, `AgentResult`, `tool_spec`. The v0.2.0 baseline at
  `tests/_api_baseline.json` is unchanged; all 27 pre-existing signatures
  remain backwards compatible.

## [0.2.0] - 2026-05-16

### Added
- `simple_agent.compute_prompt_hash(text)` — 16-char SHA-256 prefix for pinning
  a prompt alongside its hash so replay determinism is self-verifying.
- `PipelineConfig.needs_search` / `AsyncPipelineConfig.needs_search` — optional
  callable gate run before `ground_fn`. Returning `False` skips grounding.
  The async config accepts either a sync predicate or an awaitable.
- `Persona.quality_rules` — optional per-persona rule list. When set, it
  REPLACES (does not merge with) the pipeline default rules for that
  persona's drafts.
- New built-in `prompt_leak` quality rule. Catches common system-prompt
  fragments ("as an AI language model", `[INST]`, `<|im_start|>`, knowledge
  cutoff disclaimers) leaked into output.
- New default quality-rule categories: `vague_attribution`,
  `negative_parallelism`, `copula_avoidance`, `knowledge_cutoff_disclaimer`,
  `generic_positive_conclusion`.
- Opt-in `statistical_rules()` factory: `burstiness_check` and
  `causal_connector_ratio` for content-window statistics beyond regex.
- Opt-in `default_rules_de()` factory: seven generic German AI-tell rules
  (Eröffnungsformel, Schlussformel, -orientiert/-basiert Komposita,
  Genitivkette, Nominalisierungs-Häufung, Dreierregel, Anglizismus-Buzz).
- `scripts/check_dedup.py` — CI gate enforcing that sync/async function pairs
  share an extracted core. Every pair is now within a 4-line "unique body"
  budget.
- `CONTRIBUTING.md`, `SECURITY.md`, `pyrightconfig.json`.

### Changed
- `quality.py` AI-vocabulary list: 7 → 30 words sourced from the PubMed AI-tell
  overuse data.
- `em_dash` rule: now a paragraph-count threshold (≥3 dashes per paragraph)
  instead of flag-on-any. Single em dashes are normal human prose.
- `_extract_json` now also strips smart quotes (`""''`), whole-line `//`
  comments, and trailing commas before parsing. The `//` stripper deliberately
  matches only at the start of a logical line, so it never corrupts URLs or
  natural-language strings containing `//`.
- `Python 3.11+` required (was `3.10+`). 3.10 reached EOL October 2025.
- CI matrix runs against 3.11/3.12/3.13/3.14; action major versions bumped.
- Sync and async pipeline cores share extracted helpers (`_batch_iter_sync`,
  `_batch_iter_async`, `_should_skip_search`, `_async_should_skip_search`,
  `_finalize_batch`, `_chat_kwargs`, `_user_messages`,
  `_system_user_messages`, `_content_or_empty`, `_backoff_seconds`). Surface
  API is unchanged.
- Examples lifted model strings to module-level constants (`SCORER_MODEL`,
  `REASONER_MODEL`, `WRITER_MODEL`) with verification-date comments. Defaults
  updated to current OpenRouter free tier as of 2026-05-15:
  `nvidia/nemotron-nano-9b-v2`, `deepseek/deepseek-v4-flash`,
  `google/gemma-4-31b-it`.
- DDG search in `examples/` now carries a prominent "demo only" docstring
  recommending Tavily, SerpAPI, or Brave Search for production.
- README "Why this instead of…" section adds smolagents and OpenAI Agents SDK
  as comparators, plus an Anti-patterns callout.

### Removed
- Manual YAML parser fallback in `persona.py`. `pyyaml` is a declared runtime
  dependency; the fallback was unreachable and untested.
- `openai` SDK mock stub in `llm.py`. `openai` is a declared runtime
  dependency; the stub raised `ModuleNotFoundError` on use anyway.

## [0.1.0] - 2026-04-04

Initial release.
