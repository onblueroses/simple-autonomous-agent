# simple-autonomous-agent

[![CI](https://github.com/onblueroses/simple-autonomous-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/onblueroses/simple-autonomous-agent/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/onblueroses/simple-autonomous-agent)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

~1,000 lines of Python for building autonomous LLM agents that use different models for different jobs, ground themselves with real data before writing, and validate their own output.

Simple to use :).

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
    scorer=ModelConfig("google/gemma-3-12b-it:free", max_tokens=256),
    reasoner=ModelConfig("deepseek/deepseek-r1:free", max_tokens=1024),
    writer=ModelConfig("deepseek/deepseek-chat-v3-0324:free", max_tokens=1024),
    scorer_client=client,
    writer_client=client,
    score_threshold=0.6,
)

persona = load_persona("personas/analyst.yaml")
item = {"id": "1", "text": "What's a realistic rental yield in Berlin in 2026?"}

result = run_pipeline(item, config, personas=[persona])
print(result.draft)
```

Three models for three jobs: a 12B for scoring (pennies per thousand calls), a thinking model for analysis, a writer for the actual output.

## Why this instead of LangChain / CrewAI / AutoGen

Those frameworks give you an orchestration layer with dozens of abstractions, plugin systems, and hundreds of transitive dependencies. This library does one thing: run a score-ground-reason-draft pipeline with fault tolerance at every stage.

- **Two runtime dependencies** (`openai`, `pyyaml`). No dependency tree to audit.
- **Multi-model routing by design** - cheap models score, thinking models reason, writing models draft. You're not paying GPT-4 prices to decide if an item is worth processing.
- **Output quality built in** - regex rules catch AI writing patterns (em dashes, "delve/crucial/landscape", filler closings) before the output leaves the pipeline. Most frameworks treat output quality as someone else's problem.
- **Each stage fails independently** - if your search API is down, the draft still happens without grounding. Nothing cascades. This matters in production, where something is always half-broken.
- **~1,000 lines you can read in one sitting**. No framework to learn, no magic. Fork it, change it, own it.

If you need tool use, multi-agent conversations, or RAG pipelines, use a framework. If you need a reliable content pipeline that sounds human, this is smaller and more focused.

## Async

All sync functions have async counterparts with `a`-prefix naming. No new runtime dependencies - `openai>=1.0.0` ships `AsyncOpenAI` and `asyncio` is stdlib.

```python
from simple_agent import (
    AsyncPipelineConfig, ModelConfig, acreate_client,
    load_persona, arun_batch,
)

client = acreate_client("https://openrouter.ai/api/v1", "your-key")

config = AsyncPipelineConfig(
    scorer=ModelConfig("google/gemma-3-12b-it:free", max_tokens=256),
    reasoner=ModelConfig("deepseek/deepseek-r1:free", max_tokens=1024),
    writer=ModelConfig("deepseek/deepseek-chat-v3-0324:free", max_tokens=1024),
    scorer_client=client,
    writer_client=client,
)

results = await arun_batch(items, config, max_concurrency=5)
```

`arun_batch` uses `asyncio.Semaphore` for bounded concurrency instead of `time.sleep` delays. `AsyncPipelineConfig` takes `AsyncOpenAI` clients and an async `ground_fn`.

## Run the examples

```bash
export OPENROUTER_API_KEY=sk-or-v1-...   # free at openrouter.ai/keys
python examples/real_api.py              # sync
python examples/async_real_api.py        # async (3 items concurrently)
```

The sync example demonstrates the full pipeline with DuckDuckGo grounding, persona-voiced drafting, and quality validation. The async example runs a batch of 3 items concurrently.

## Install

```bash
git clone https://github.com/onblueroses/simple-autonomous-agent.git
cd simple-autonomous-agent
pip install -e ".[dev]"
```

Two dependencies: `openai` and `pyyaml`.

## Modules

**`llm.py`** - `create_client()`, `score()`, `reason()`, `draft()` and their async counterparts (`acreate_client()`, `ascore()`, `areason()`, `adraft()`). Each function targets a different job. `score()` handles cheap classification, `reason()` handles thinking models that return output in `reasoning`, `reasoning_content`, or `<think>` tags. `draft()` uses system-message identity framing for persona voice. All calls include retry with exponential backoff for rate limits and timeouts.

**`pipeline.py`** - `run_pipeline()` wires the stages together with try/except around each one. JSON extraction handles LLM responses wrapped in markdown fences or preamble text. `run_batch()` adds rate limiting and run logging. Async versions `arun_pipeline()` and `arun_batch()` provide the same behavior with semaphore-bounded concurrency.

**`persona.py`** - Loads YAML persona configs. `build_system_prompt()` frames the persona as identity ("You are Marcus Voss...") rather than instruction ("Write like an analyst"). The identity framing produces better voice consistency.

**`quality.py`** - 8 regex rules for catching AI writing patterns (em dashes, "delve/crucial/landscape", three-point lists, filler closings). Also has `sanitize_input()` for stripping prompt injection from untrusted text. The defaults are a starting point - you'll want to tune them.

**`state.py`** - SQLite wrapper. Three tables: items (deduplication), drafts (lifecycle), runs (logging). Supports in-memory for testing.

**`config.py`** - Dataclasses. `ModelConfig`, `PipelineConfig`, `AsyncPipelineConfig`, `PipelineResult`. No env vars, no global state. Prompt templates, retry parameters, and score thresholds are all configurable.

## Tests

```bash
python -m pytest tests/ -v
```

96 tests, no API keys needed. Pipeline tests use mocked LLM calls.

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

## License

Apache 2.0
