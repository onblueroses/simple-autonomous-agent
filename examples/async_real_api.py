"""Run the async agent pipeline with real LLM calls via OpenRouter.

Setup:
    1. Get a free API key at https://openrouter.ai/keys (no credit card needed)
    2. Set it: export OPENROUTER_API_KEY=sk-or-v1-...
    3. Run: python examples/async_real_api.py

Demonstrates concurrent batch processing: 3 items run in parallel with
max_concurrency=3 instead of sequential processing with delays.
"""

import asyncio
import os
import re
import sys
import urllib.parse
import urllib.request

# Add parent dir to path so this runs from the examples/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simple_agent import (
    AsyncPipelineConfig,
    ModelConfig,
    acreate_client,
    arun_batch,
    default_rules,
    load_persona,
)


# --- Grounding: DuckDuckGo search wrapped for async ---

def _ddg_search_sync(query: str) -> str:
    """Blocking DuckDuckGo search. Called via asyncio.to_thread."""
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"Search failed: {e}"

    snippets = re.findall(r'class="result__snippet">(.*?)</a>', html, re.DOTALL)
    if not snippets:
        return "No search results found."
    clean = [re.sub(r"<[^>]+>", "", s).strip() for s in snippets[:5]]
    return "\n".join(f"- {s}" for s in clean if s)


async def ddg_search(query: str) -> str:
    """Async wrapper around blocking DuckDuckGo search."""
    return await asyncio.to_thread(_ddg_search_sync, query)


async def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY to run this example.")
        print("Get a free key at: https://openrouter.ai/keys")
        sys.exit(1)

    client = acreate_client("https://openrouter.ai/api/v1", api_key)
    persona = load_persona(os.path.join(os.path.dirname(__file__), "..", "personas", "analyst.yaml"))

    config = AsyncPipelineConfig(
        scorer=ModelConfig("google/gemma-3-12b-it:free", max_tokens=256),
        reasoner=ModelConfig("deepseek/deepseek-r1:free", max_tokens=1024),
        writer=ModelConfig("deepseek/deepseek-chat-v3-0324:free", max_tokens=1024),
        scorer_client=client,
        writer_client=client,
        ground_fn=ddg_search,
        quality_rules=default_rules(),
    )

    items = [
        {"id": "demo-1", "text": "What's a realistic rental yield in Berlin in 2026?"},
        {"id": "demo-2", "text": "How does Denkmal-AfA work for heritage buildings?"},
        {"id": "demo-3", "text": "Is Pflegeimmobilien a good investment for passive income?"},
    ]

    print(f"Running async batch pipeline ({len(items)} items, max_concurrency=3)...\n")
    results = await arun_batch(items, config, personas=[persona], max_concurrency=3)

    for result in results:
        print(f"\n{'='*60}")
        print(f"Item: {result.item_id}")
        print(f"Score: {result.score}")
        print(f"Persona: {result.persona}")
        print(f"Quality passed: {result.passed_quality}")
        if result.draft:
            print(f"\n--- Draft ---\n{result.draft[:300]}...")
        if result.errors:
            print(f"\n--- Errors ---")
            for e in result.errors:
                print(f"  {e}")


if __name__ == "__main__":
    asyncio.run(main())
