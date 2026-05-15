# Contributing

Thanks for considering a contribution. This library is small on purpose;
please keep it that way.

## Dev setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[dev]
```

## Running the test suite

```bash
pytest -q
```

The suite uses mocks for every LLM call; no API keys are needed.

## Type and dedup gates

```bash
pyright simple_agent/
python scripts/check_dedup.py
```

`check_dedup.py` enforces that every sync/async function pair shares an
extracted core. New code that adds a sync/async pair must keep the
non-shared body within the budget (`MAX_UNIQUE_LINES`).

## PR expectations

- One logical change per PR. If the work touches multiple concerns,
  split it.
- One test per behavior change. Bug fixes ship with a regression test.
- No new runtime dependencies without an issue discussion first. The
  current dependencies are `openai` and `pyyaml`, period.
- No defensive code for scenarios that can't happen. Trust the declared
  dependencies. Validate at system boundaries only.
- Comments explain WHY, not WHAT. The reader knows Python.
- Commit messages: subject-only, present tense, under 70 chars; describe
  the WHY in the PR body if needed.

## Out of scope

This library does NOT do tool calls, multi-agent conversation, or RAG.
Those belong in a different project. Please don't add them.
