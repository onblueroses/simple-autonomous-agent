# simple-autonomous-agent

[![CI](https://github.com/onblueroses/simple-autonomous-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/onblueroses/simple-autonomous-agent/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/onblueroses/simple-autonomous-agent)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

~1,000 lines of Python for building autonomous LLM agents that use different models for different jobs, ground themselves with real data before writing, and validate their own output.

Simple to use.

## How it works

```
Input item
    |
    v
[Score]          cheap model decides if the item is worth processing
    |
    v
[Select persona]  picks the right voice for the content
    |
    v
[Ground]         calls your search function for real-world context (optional)
    |
    v
[Reason]         thinking model analyzes the content + grounding
    |
    v
[Draft]          writing model generates output in persona voice
    |
    v
[Validate]       regex rules catch AI writing patterns in the output
    |
    v
[Persist]        SQLite state store for deduplication and run logs (optional)
```

Each stage fails independently. If search is down, the draft still happens - just without grounding context. If the reasoning model times out, the writer works without the analysis. Nothing cascades.

## Quick start

```python
from simple_agent import (
    ModelConfig, PipelineConfig, create_client,
    load_persona, run_pipeline,
)

client = create_client("https://openrouter.ai/api/v1", "your-key")

config = PipelineConfig(
    scorer=ModelConfig("nvidia/nemotron-nano-9b-v2:free", max_tokens=256),
    reasoner=ModelConfig("deepseek/deepseek-v4-flash:free", max_tokens=1024),
    writer=ModelConfig("google/gemma-4-31b-it:free", max_tokens=1024),
    scorer_client=client,
    writer_client=client,
    score_threshold=0.6,
)

persona = load_persona("personas/analyst.yaml")
item = {"id": "1", "text": "How do open-source dependency audit tools compare on signal-to-noise?"}

result = run_pipeline(item, config, personas=[persona])
print(result.draft)
```

Three models for three jobs: a small one for scoring (pennies per thousand calls), a long-context model for analysis, a stronger one for the actual draft. Model IDs are OpenRouter free-tier as of 2026-05-15; swap for any OpenAI-compatible provider.

## Why this instead of LangChain / CrewAI / AutoGen / smolagents / OpenAI Agents SDK

Those frameworks give you an orchestration layer with dozens of abstractions, plugin systems, and hundreds of transitive dependencies. The newer minimalist contenders (smolagents, OpenAI Agents SDK) are closer in spirit but still pull in their own runtime concerns: tool registries, handoff primitives, tracing infrastructure. This library does one thing: run a score-ground-reason-draft pipeline with fault tolerance at every stage.

- **Two runtime dependencies** (`openai`, `pyyaml`). No dependency tree to audit.
- **Multi-model routing by design** - cheap models score, thinking models reason, writing models draft. You're not paying GPT-4 prices to decide if an item is worth processing.
- **Output quality built in** - 13 regex rules catch AI writing patterns (em dashes, the `delve/crucial/landscape/showcase/tapestry` vocabulary, vague attributions, negative parallelism, knowledge-cutoff disclaimers, prompt-leak fragments) before the output leaves the pipeline.
- **Each stage fails independently** - if your search API is down, the draft still happens without grounding. Nothing cascades. This matters in production, where something is always half-broken.
- **Cost-gated grounding** - optional `needs_search` predicate runs before `ground_fn`, so you don't pay for a search call when the question doesn't need fresh facts.
- **Prompt versioning** - `compute_prompt_hash(text)` returns a 16-char SHA-256 prefix you can pin alongside the prompt itself, so replay determinism is self-verifying.

If you need tool calls, multi-agent conversations, or RAG pipelines, use a framework. If you need a reliable content pipeline that sounds human, this is smaller and more focused.

### Anti-patterns this library is opinionated against

- **Don't fan the same request to N models in parallel and pick the best.** Wastes tokens, creates ambiguous results. Use the score-then-route pattern instead.
- **Don't run tool loops without an iteration budget.** Runaway agents are real. The library deliberately doesn't expose a tool-loop primitive; if you build one on top, cap it.
- **Don't mock dependencies with stubs that just raise.** `import openai` already raises informatively when openai isn't installed. The same applies to your library: lean on declared dependencies, don't paper over missing ones.
- **Don't write defensive code for scenarios that can't happen.** Validate at system boundaries (user input, network responses). Trust types internally.

## Async

All sync functions have async counterparts with `a`-prefix naming. No new runtime dependencies - `openai>=1.0.0` ships `AsyncOpenAI` and `asyncio` is stdlib.

```python
from simple_agent import (
    AsyncPipelineConfig, ModelConfig, acreate_client,
    load_persona, arun_batch,
)

client = acreate_client("https://openrouter.ai/api/v1", "your-key")

config = AsyncPipelineConfig(
    scorer=ModelConfig("nvidia/nemotron-nano-9b-v2:free", max_tokens=256),
    reasoner=ModelConfig("deepseek/deepseek-v4-flash:free", max_tokens=1024),
    writer=ModelConfig("google/gemma-4-31b-it:free", max_tokens=1024),
    scorer_client=client,
    writer_client=client,
)

results = await arun_batch(items, config, max_concurrency=5)
```

`arun_batch` uses `asyncio.Semaphore` for bounded concurrency instead of `time.sleep` delays. `AsyncPipelineConfig` takes `AsyncOpenAI` clients and an async `ground_fn`.

## Run the examples

```bash
export OPENROUTER_API_KEY="<your-openrouter-key>"   # free at openrouter.ai/keys
python examples/real_api.py              # sync
python examples/async_real_api.py        # async (3 items concurrently)
```

The sync example demonstrates the full pipeline with DuckDuckGo grounding, persona-voiced drafting, and quality validation. The async example runs a batch of 3 items concurrently.

## Modules

**`llm.py`** - `create_client()`, `score()`, `reason()`, `draft()` and their async counterparts (`acreate_client()`, `ascore()`, `areason()`, `adraft()`). Each function targets a different job. `score()` handles cheap classification, `reason()` handles thinking models that return output in `reasoning`, `reasoning_content`, or `<think>` tags. `draft()` uses system-message identity framing for persona voice. All calls include retry with exponential backoff for rate limits and timeouts.

**`pipeline.py`** - `run_pipeline()` wires the stages together with try/except around each one. JSON extraction handles LLM responses wrapped in markdown fences or preamble text. `run_batch()` adds rate limiting and run logging. Async versions `arun_pipeline()` and `arun_batch()` provide the same behavior with semaphore-bounded concurrency.

**`persona.py`** - Loads YAML persona configs. `build_system_prompt()` frames the persona as identity ("You are Marcus Voss...") rather than instruction ("Write like an analyst"). The identity framing produces better voice consistency.

**`quality.py`** - 13 default regex rules for AI writing patterns: em-dash count threshold, ~30-word AI-vocabulary list, three-point lists, filler openings/closings, hedge piles, excessive transitions, vague attributions, negative parallelism, copula avoidance, knowledge-cutoff disclaimers, generic positive conclusions, and a `prompt_leak` rule that catches system-prompt fragments leaking through to output. Also has `sanitize_input()` for stripping prompt injection from untrusted text. `check_quality()` and `validate_output()` return concrete violations. The defaults are a starting point - tune for your domain.

**`state.py`** - SQLite wrapper. Three tables: items (deduplication), drafts (lifecycle), runs (logging). The `StateStore` is a context manager (`with StateStore(path) as state:`), so connections close cleanly. Supports `":memory:"` for testing.

**`versioning.py`** - `compute_prompt_hash(text)` returns a 16-char SHA-256 prefix. Pin a prompt's hash next to the prompt and you have replay-determinism that survives refactors.

**`config.py`** - Dataclasses. `ModelConfig`, `PipelineConfig`, `AsyncPipelineConfig`, `PipelineResult`. No env vars, no global state. Prompt templates, retry parameters, and score thresholds are all configurable.

## Tests

```bash
python -m pytest tests/ -v
```

108 tests, no API keys needed. Pipeline tests use mocked LLM calls.

## Configuring prompts

The scorer and persona selection prompts are configurable via `PipelineConfig`:

```python
config = PipelineConfig(
    ...,
    scorer_prompt_template=(
        "Is this content about finance? Rate 0.0 to 1.0.\n"
        'Return JSON: {"score": <float>}\n\n{content}'
    ),
)
```

Use `{content}` as placeholder for the input text, `{personas}` for persona names in the selection prompt. Defaults reproduce the built-in prompts.

## Retry and robustness

LLM calls retry automatically on rate limits (429), timeouts, and connection errors. Default: 2 retries with exponential backoff (1s, 2s). Configure via `PipelineConfig`:

```python
config = PipelineConfig(
    ...,
    max_retries=3,        # 0 to disable
    retry_base_delay=0.5, # seconds
)
```

Score parsing handles JSON wrapped in markdown fences or preceded by preamble text. Thinking models are supported across three provider patterns (`reasoning_content`, `reasoning` fields, and `<think>` tags).

## Personas

```yaml
name: "Your Persona Name"
identity: >
  Background and perspective, written as if describing who this
  person IS rather than what they should do.

voice: >
  How they communicate. Sentence structure, formality, patterns.

expertise:
  - Domain 1
  - Domain 2

constraints:
  - Style rules specific to this persona
  - Quality standards for their output

example_outputs:
  - >
    A sample of how this persona writes. Concrete details,
    numbers, specifics - not vague descriptions.
```

## Custom quality rules

```python
from simple_agent import QualityRule, default_rules

my_rules = default_rules() + [
    QualityRule(
        name="no_jargon",
        pattern=r"\b(?:synergize|leverage|ideate)\b",
        description="Corporate jargon.",
    ),
]

config = PipelineConfig(..., quality_rules=my_rules)
```

You can also attach rules to a persona, in which case they **replace** the
pipeline default rules for that persona's drafts:

```python
from simple_agent import Persona, QualityRule, run_pipeline

persona = load_persona("personas/legal.yaml")
persona.quality_rules = [
    QualityRule(name="no_em_dash", pattern=r"\u2014", description="In-house style."),
    QualityRule(name="no_first_person", pattern=r"\bI\b", description="Memo voice."),
]
```

## Cost-gated grounding

`ground_fn` calls can be the most expensive step in the pipeline. Skip them
when the question doesn't need fresh data:

```python
def needs_search(item, persona, config) -> bool:
    # Cheap-model call or domain heuristic
    return "today" in item["text"].lower() or "current" in item["text"].lower()

config = PipelineConfig(..., ground_fn=tavily_search, needs_search=needs_search)
```

The async config accepts either a sync predicate or one returning an awaitable.

## Prompt versioning

```python
from simple_agent import compute_prompt_hash

PROMPT_V1 = "Classify the sentiment of this content: ..."
PROMPT_V1_HASH = "b83b31b8ffa9742c"
assert compute_prompt_hash(PROMPT_V1) == PROMPT_V1_HASH
```

Replays of stored prompt logs are now self-verifying.

## License

Apache 2.0
