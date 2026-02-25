# simple-autonomous-agent

Multi-model LLM orchestration and output quality engineering in ~700 lines of Python.

You don't need LangChain, CrewAI, or AutoGen. Most autonomous agents follow the same pattern: score incoming content, ground yourself with real data, reason about it with a thinking model, draft a response with a writing model, validate the output isn't AI slop. This library implements that pattern with zero framework overhead.

This isn't a framework you configure - it's a pattern you read and adapt. Each module is independent, every function has type hints, and the whole thing fits in your head in 20 minutes.

## Architecture

```
Input items
    |
    v
[Score] -----> below threshold? -> skip
    |           (cheap, fast model)
    v
[Select Persona] -> match item to the right voice
    |
    v
[Ground] -----> search for real data (optional, your function)
    |            failure here doesn't kill the pipeline
    v
[Reason] -----> thinking model analyzes content + grounding
    |            failure here doesn't kill the pipeline
    v
[Draft] ------> writing model generates output with persona identity
    |            grounding + reasoning injected as XML context blocks
    v
[Validate] ---> quality rules catch AI slop patterns
    |
    v
[Persist] ----> SQLite state store (optional)
    |
    v
PipelineResult (score, grounding, reasoning, draft, errors)
```

**Key design choice:** Every stage wraps in try/except. A failed search doesn't prevent drafting - it just means the draft won't have external context. A failed reasoning step means the writer works without analysis. The pipeline degrades gracefully instead of crashing.

## Why this exists

The agent framework landscape has two modes:

1. **Toy examples** - "call GPT-4 and print the response" wrapped in 15 files of abstraction
2. **Massive frameworks** - 50,000 lines, plugin registries, YAML DSLs, and a PhD's worth of concepts to learn before you can write hello world

This sits in the middle: production-tested patterns extracted from a real autonomous agent, small enough to read end-to-end, zero magic.

## Quick start

```python
from simple_agent import (
    ModelConfig, PipelineConfig, create_client,
    load_persona, run_pipeline,
)

client = create_client("https://openrouter.ai/api/v1", "your-key")

config = PipelineConfig(
    scorer=ModelConfig("google/gemma-3-12b-it:free", max_tokens=256),
    reasoner=ModelConfig("qwen/qwen3-235b-a22b:free", max_tokens=1024),
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

Three models, three jobs. Scoring uses a cheap 12B model (pennies per thousand calls). Reasoning uses a thinking model that's good at analysis. Writing uses a model that's good at natural language. Each does what it's best at.

## Installation

```bash
git clone https://github.com/your-username/simple-autonomous-agent.git
cd simple-autonomous-agent
pip install -e ".[dev]"
```

Dependencies: `openai` (for any OpenAI-compatible API) and `pyyaml` (for persona configs). That's it.

## Modules

### `llm.py` - Model routing

Four functions: `create_client()`, `score()`, `reason()`, `draft()`. Each wraps the OpenAI chat API for its specific job. The `reason()` function handles thinking models that put output in a `reasoning` field instead of `content` - you don't have to think about that.

### `pipeline.py` - Orchestration

`run_pipeline()` wires the seven stages together. `run_batch()` iterates over multiple items with rate limiting and run logging. Fault tolerance is built in - each stage fails independently.

### `persona.py` - Identity, not instructions

Personas are YAML configs with structured fields: name, identity, voice, expertise, constraints, example outputs. `build_system_prompt()` turns a persona into a system message that frames the persona as who you ARE, not as instructions to follow. This produces more consistent voice than "write like an analyst."

### `quality.py` - Fighting AI slop

`default_rules()` ships 8 regex-backed quality rules that catch the most recognizable AI writing patterns: em dashes, "delve/crucial/landscape," exactly-three-point lists, filler openings ("That's a great question"), filler closings ("I hope this helps"). These are a starting point - configure them for your domain.

Also includes `sanitize_input()` for stripping prompt injection patterns from untrusted input before it reaches the LLM.

### `state.py` - Persistence

`StateStore` wraps SQLite with three tables: items (deduplication), drafts (lifecycle management), and runs (observability). In-memory mode (`:memory:`) for testing, file-backed for production. Includes draft expiry for stale items.

### `config.py` - Types

Plain dataclasses: `ModelConfig`, `PipelineConfig`, `PipelineResult`. No env var loading, no global state, no magic. You instantiate them and pass them to functions.

## What makes this different

**Multi-model routing.** Most frameworks use one model for everything. Production agents route different tasks to different models - a 12B model for yes/no scoring, a 235B thinking model for analysis, a tuned model for writing. This library makes that pattern explicit.

**Grounding before generation.** The `ground_fn` parameter accepts any function that takes a query and returns context text. Search APIs, database lookups, file reads - whatever grounds your agent in reality. This runs before the reasoning step, so the thinking model works with real data, not hallucinations.

**Separation of reasoning and writing.** The reasoning model produces structured analysis. The writing model takes that analysis and produces output in a persona's voice. Each model does what it's best at instead of one model trying to do both.

**Output quality validation.** Most agent frameworks treat output as a black box. This library validates output against configurable rules before returning it. The default rules target the most common AI writing patterns - the ones that make readers think "a chatbot wrote this."

**Graceful degradation.** Every stage wraps in try/except. Failed grounding doesn't block drafting. Failed reasoning doesn't block writing. The pipeline produces the best output it can with whatever stages succeed. Production systems need this - APIs go down, models timeout, rate limits hit.

**Persona internalization.** System prompts frame the persona as identity ("You are Marcus Voss. An investment analyst with 12 years of experience...") not as instruction ("Write like an investment analyst"). This produces more consistent, natural voice because the model adopts the identity rather than following formatting rules.

## Running tests

```bash
python -m pytest tests/ -v
```

48 tests covering quality rules, state management, pipeline orchestration (with mocked LLM calls), and persona loading. All tests run without API keys.

## Custom personas

Create a YAML file in `personas/`:

```yaml
name: "Your Persona Name"
identity: >
  Background, experience, and perspective. Written as if describing
  who this person IS, not what they should do.

voice: >
  How they communicate. Sentence structure, formality level,
  distinctive patterns.

expertise:
  - Domain 1
  - Domain 2

constraints:
  - Rules the persona always follows
  - Quality standards specific to this voice

example_outputs:
  - >
    A sample of how this persona actually writes.
    Include concrete details - numbers, comparisons, specifics.
```

## Custom quality rules

```python
from simple_agent import QualityRule, default_rules

my_rules = default_rules() + [
    QualityRule(
        name="no_jargon",
        pattern=r"\b(?:synergize|leverage|ideate)\b",
        description="Corporate jargon that alienates readers.",
    ),
]

config = PipelineConfig(..., quality_rules=my_rules)
```

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
