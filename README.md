# simple-autonomous-agent

[![CI](https://github.com/onblueroses/simple-autonomous-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/onblueroses/simple-autonomous-agent/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/onblueroses/simple-autonomous-agent)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A minimal `Agent` that wraps any OpenAI-compatible model with callable tools and a bounded loop, plus a specialized score-ground-reason-draft pipeline for content workflows that need to sound human.

Two runtime dependencies (`openai`, `pyyaml`). ~1,300 lines of Python.

## Quick start: Agent

```python
from simple_agent import Agent, create_client

def calculate(expression: str) -> str:
    """Evaluate a numeric arithmetic expression."""
    import ast, operator
    ops = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv}
    def _eval(node):
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.BinOp): return ops[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError("disallowed")
    return str(_eval(ast.parse(expression, mode="eval").body))

def lookup_constant(name: str) -> str:
    """Return a known mathematical or physical constant."""
    return {"pi": "3.14159", "speed_of_light": "2.998e8 m/s"}.get(name, "unknown")

client = create_client("https://openrouter.ai/api/v1", "your-key")
agent = Agent(
    client=client,
    model="google/gemma-4-31b-it:free",
    tools=[calculate, lookup_constant],
    system_prompt="Use tools when you need to compute or look up values.",
    max_steps=6,
)

result = agent.run("What is pi times 1000?")
print(result.output)        # "3141.59"
print(result.steps)         # 2
print(result.terminated)    # "done"
```

Tools are plain functions with type hints. The agent auto-generates the OpenAI tool spec, dispatches calls, feeds results back to the model, and loops until the model produces a final answer or `max_steps` is hit. There is no decorator, no registry, no plugin system.

For async clients, use `AsyncAgent.arun(task)` with the same constructor shape.

## Why this instead of LangChain / CrewAI / AutoGen / smolagents / OpenAI Agents SDK

Those frameworks give you an orchestration layer with dozens of abstractions, plugin systems, and hundreds of transitive dependencies. The newer minimalist contenders (smolagents, OpenAI Agents SDK) are closer in spirit but still pull in their own runtime concerns: tool registries, handoff primitives, tracing infrastructure. This library has two front doors and nothing else: an `Agent` for tool-calling loops, and a `run_pipeline` for score-ground-reason-draft content workflows.

- **Two runtime dependencies** (`openai`, `pyyaml`). No dependency tree to audit.
- **Two front doors, both small** - `Agent` is a bounded tool loop; `run_pipeline` is a multi-model content pipeline. Pick one, ignore the other, or use both. They don't depend on each other.
- **Multi-model routing by design** - cheap models score, thinking models reason, writing models draft. You're not paying GPT-4 prices to decide if an item is worth processing.
- **Output quality built in** - 13 regex rules catch AI writing patterns (em dashes, the `delve/crucial/landscape/showcase/tapestry` vocabulary, vague attributions, negative parallelism, knowledge-cutoff disclaimers, prompt-leak fragments) before the output leaves the pipeline.
- **Each stage fails independently** - if your search API is down, the draft still happens without grounding. Nothing cascades.
- **Cost-gated grounding** - optional `needs_search` predicate runs before `ground_fn`, so you don't pay for a search call when the question doesn't need fresh facts.
- **Prompt versioning** - `compute_prompt_hash(text)` returns a 16-char SHA-256 prefix you can pin alongside the prompt itself.

## Agent in depth

### Tools

Tools are callables in v0.3.0. The agent inspects each function's signature and type hints to build the OpenAI tool spec automatically:

```python
def search(query: str, limit: int = 5) -> str: ...
def lookup(name: str | None = None) -> str: ...

agent = Agent(client=client, model="...", tools=[search, lookup])
```

Supported parameter types: `str`, `int`, `float`, `bool`, `list[T]`, `Optional[T]`, `T | None`. A parameter is required iff it has no default value and its annotation is not optional. Untyped or unsupported-type parameters raise `TypeError` at agent construction.

Raw dict tool specs are deferred to a later release.

### Termination

`AgentResult.terminated` is one of three strings:

- `"done"` - the model returned content with no tool calls. `result.output` is that content.
- `"max_steps"` - the loop budget was exhausted. `result.output` is the last model content (may be empty).
- `"no_progress"` - the model returned neither content nor tool calls. `result.output` is `""`.

`result.steps` counts model calls. One assistant response = one step regardless of how many tool calls it emits in parallel.

### Memory

Memory is caller-owned. Pass a list and the agent appends to it in place; the same list is returned as `result.messages`:

```python
mem: list[dict] = []
agent.run("first question", memory=mem)
agent.run("follow-up", memory=mem)       # same conversation
```

The system message (if `system_prompt` is set) is inserted at index 0 only when memory is empty or doesn't already start with a system message. Repeated runs against the same memory don't duplicate it.

### Tool errors are observations

A tool that raises an exception doesn't bubble up. The exception is formatted into a tool message (`"Error: TypeError('bad arg')"`) so the model can read it and try again within the remaining budget. If you want a tool's exception to halt the agent, catch it inside the tool and re-raise outside.

### Retries

LLM calls retry on `RateLimitError`, `APITimeoutError`, and `APIConnectionError` with exponential backoff. Defaults: 2 retries, 1.0s base delay. Configure via `max_retries=` and `retry_base_delay=` on the `Agent` constructor.

## Anti-patterns

- **Don't fan the same request to N models in parallel and pick the best.** Wastes tokens, creates ambiguous results. Use the score-then-route pattern instead.
- **Don't run tool loops without an iteration budget.** Runaway agents are real. `Agent` enforces `max_steps >= 1` at construction; raise it knowingly.
- **Don't mock dependencies with stubs that just raise.** `import openai` already raises informatively when openai isn't installed. The same applies to your library: lean on declared dependencies, don't paper over missing ones.
- **Don't write defensive code for scenarios that can't happen.** Validate at system boundaries (user input, network responses). Trust types internally.

## Async

`AsyncAgent` mirrors `Agent` with `await agent.arun(task)`. Tools may be `async def`, `functools.partial` around an async function, or sync functions; awaitable results are detected and awaited automatically.

```python
from simple_agent import AsyncAgent, acreate_client

client = acreate_client("https://openrouter.ai/api/v1", "your-key")

async def fetch_price(symbol: str) -> str:
    """Look up a current stock price."""
    ...  # your async HTTP call

agent = AsyncAgent(client=client, model="...", tools=[fetch_price])
result = await agent.arun("What's TSLA trading at?")
```

The content pipeline has async counterparts too: `arun_pipeline`, `arun_batch`, `ascore`, `areason`, `adraft`, `acreate_client`. `arun_batch` uses `asyncio.Semaphore` for bounded concurrency instead of `time.sleep`.

## Specialized: Content Pipeline

When you need score-ground-reason-draft-validate-persist instead of an open-ended tool loop, the pipeline is a specialized front door:

```
Input item
    |
    v
[Score]          cheap model decides if the item is worth processing
    |
    v
[Select persona] picks the right voice for the content
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

### Quick start: pipeline

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

### Modules

**`llm.py`** - `create_client()`, `score()`, `reason()`, `draft()` and their async counterparts. `score()` handles cheap classification, `reason()` handles thinking models that return output in `reasoning`, `reasoning_content`, or `<think>` tags. `draft()` uses system-message identity framing for persona voice. All calls include retry with exponential backoff for rate limits and timeouts.

**`pipeline.py`** - `run_pipeline()` wires the stages together with try/except around each one. JSON extraction handles LLM responses wrapped in markdown fences or preamble text. `run_batch()` adds rate limiting and run logging. Async versions provide the same behavior with semaphore-bounded concurrency.

**`agent.py`** - `Agent`, `AsyncAgent`, `AgentResult`, `tool_spec`. Bounded tool-calling loop with auto-derived schemas. Decoupled from the pipeline; the agent works against any model + callable tools.

**`persona.py`** - Loads YAML persona configs. `build_system_prompt()` frames the persona as identity ("You are Marcus Voss...") rather than instruction. The identity framing produces better voice consistency.

**`quality.py`** - 13 default regex rules for AI writing patterns. Also has `sanitize_input()` for stripping prompt injection from untrusted text. The defaults are a starting point - tune for your domain.

**`state.py`** - SQLite wrapper. Three tables: items (deduplication), drafts (lifecycle), runs (logging). Context-manager support so connections close cleanly.

**`versioning.py`** - `compute_prompt_hash(text)` returns a 16-char SHA-256 prefix for replay-determinism.

**`config.py`** - Dataclasses. No env vars, no global state.

### Personas

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

example_outputs:
  - >
    A sample of how this persona writes.
```

### Cost-gated grounding

`ground_fn` calls can be the most expensive step. Skip them when the question doesn't need fresh data:

```python
def needs_search(item, persona, config) -> bool:
    return "today" in item["text"].lower() or "current" in item["text"].lower()

config = PipelineConfig(..., ground_fn=tavily_search, needs_search=needs_search)
```

The async config accepts either a sync predicate or one returning an awaitable.

### Custom quality rules

```python
from simple_agent import QualityRule, default_rules

my_rules = default_rules() + [
    QualityRule(name="no_jargon", pattern=r"\b(?:synergize|leverage|ideate)\b",
                description="Corporate jargon."),
]

config = PipelineConfig(..., quality_rules=my_rules)
```

Attach rules to a persona to **replace** the pipeline defaults for that persona's drafts:

```python
persona.quality_rules = [
    QualityRule(name="no_em_dash", pattern=r"\u2014", description="In-house style."),
]
```

### Prompt versioning

```python
from simple_agent import compute_prompt_hash

PROMPT_V1 = "Classify the sentiment of this content: ..."
PROMPT_V1_HASH = compute_prompt_hash(PROMPT_V1)  # 16-char SHA-256 prefix
```

Pin a hash next to the prompt and replays of stored prompt logs become self-verifying.

## Run the examples

```bash
export OPENROUTER_API_KEY="<your-openrouter-key>"   # free at openrouter.ai/keys
python examples/agent_real_api.py        # Agent loop with calculator + constants
python examples/real_api.py              # full content pipeline
python examples/async_real_api.py        # async pipeline, 3 items concurrently
```

## Tests

```bash
python -m pytest tests/ -v
```

206 tests, no API keys needed. All LLM calls mocked.

## License

Apache 2.0
