# simple-autonomous-agent

~700 lines of Python for wrapping multiple autonomous LLM agents that use different models for different jobs, ground themselves with real data before writing, and validate their own output.

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

Three models for three jobs: a 12B for scoring (pennies per thousand calls), a 235B thinker for analysis, a writer for the actual output. Lots of free models for this currently on OpenRouter.

## Install

```bash
git clone https://github.com/onblueroses/simple-autonomous-agent.git
cd simple-autonomous-agent
pip install -e ".[dev]"
```

Two dependencies: `openai` and `pyyaml`.

## Modules

**`llm.py`** - `create_client()`, `score()`, `reason()`, `draft()`. Thin wrappers around the OpenAI chat API. `reason()` handles thinking models that return output in a `reasoning` field instead of `content`.

**`pipeline.py`** - `run_pipeline()` wires the stages together with try/except around each one. `run_batch()` adds rate limiting and run logging.

**`persona.py`** - Loads YAML persona configs. `build_system_prompt()` frames the persona as identity ("You are Marcus Voss...") rather than instruction ("Write like an analyst"). The identity framing produces better voice consistency.

**`quality.py`** - 8 regex rules for catching AI writing patterns (em dashes, "delve/crucial/landscape", three-point lists, filler closings). Also has `sanitize_input()` for stripping prompt injection from untrusted text. The defaults are a starting point - you'll want to tune them.

**`state.py`** - SQLite wrapper. Three tables: items (deduplication), drafts (lifecycle), runs (logging). Supports in-memory for testing.

**`config.py`** - Dataclasses. `ModelConfig`, `PipelineConfig`, `PipelineResult`. No env vars, no global state.

## Tests

```bash
python -m pytest tests/ -v
```

48 tests, no API keys needed. Pipeline tests use mocked LLM calls.

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
