"""Run the full agent pipeline with real LLM calls via OpenRouter.

Setup:
    1. Get a free API key at https://openrouter.ai/keys (no credit card needed)
    2. Set it: export OPENROUTER_API_KEY=sk-or-v1-...
    3. Run: python examples/real_api.py

Uses three free models for three jobs:
    - Scorer: google/gemma-3-12b-it:free (fast, cheap classification)
    - Reasoner: deepseek/deepseek-r1:free (thinking model for analysis)
    - Writer: deepseek/deepseek-chat-v3-0324:free (creative, persona-voiced output)
"""

import os
import re
import sys
import urllib.request
import urllib.parse

# Add parent dir to path so this runs from the examples/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simple_agent import (
    ModelConfig,
    PipelineConfig,
    check_quality,
    create_client,
    default_rules,
    load_persona,
    run_pipeline,
)


# --- Grounding: DuckDuckGo search (no API key, stdlib only) ---

def ddg_search(query: str) -> str:
    """Search DuckDuckGo and return text snippets. Zero dependencies beyond stdlib.

    This is a basic implementation for demonstration. For production use,
    consider Tavily (see comment block below) or another search API.
    """
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"Search failed: {e}"

    # Extract result snippets from DDG HTML response
    snippets = re.findall(r'class="result__snippet">(.*?)</a>', html, re.DOTALL)
    if not snippets:
        return "No search results found."
    # Clean HTML tags from snippets
    clean = [re.sub(r"<[^>]+>", "", s).strip() for s in snippets[:5]]
    return "\n".join(f"- {s}" for s in clean if s)


# --- Tavily alternative (uncomment if you have a Tavily API key) ---
#
# from tavily import TavilyClient
#
# tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
#
# def tavily_search(query: str) -> str:
#     results = tavily.search(query, max_results=5)
#     return "\n".join(
#         f"- {r['content']}" for r in results.get("results", [])
#     )


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY to run this example.")
        print("Get a free key at: https://openrouter.ai/keys")
        sys.exit(1)

    client = create_client("https://openrouter.ai/api/v1", api_key)
    persona = load_persona(os.path.join(os.path.dirname(__file__), "..", "personas", "analyst.yaml"))

    config = PipelineConfig(
        scorer=ModelConfig("google/gemma-3-12b-it:free", max_tokens=256),
        reasoner=ModelConfig("deepseek/deepseek-r1:free", max_tokens=1024),
        writer=ModelConfig("deepseek/deepseek-chat-v3-0324:free", max_tokens=1024),
        scorer_client=client,
        writer_client=client,
        ground_fn=ddg_search,  # swap with tavily_search for better results
        quality_rules=default_rules(),
    )

    item = {"id": "demo-1", "text": "What's a realistic rental yield in Berlin in 2026?"}

    print("Running pipeline...\n")
    result = run_pipeline(item, config, personas=[persona])

    print(f"Score: {result.score}")
    print(f"Persona: {result.persona}")
    print(f"Quality passed: {result.passed_quality}")
    if result.grounding:
        print(f"\n--- Grounding ---\n{result.grounding[:500]}")
    if result.reasoning:
        print(f"\n--- Reasoning ---\n{result.reasoning[:500]}")
    if result.draft:
        print(f"\n--- Draft ---\n{result.draft}")
    if result.errors:
        print(f"\n--- Errors ---")
        for e in result.errors:
            print(f"  {e}")

    # --- Quality rules demo ---
    print("\n\n=== Quality Rules Demo ===")
    print("Running quality checks on a deliberately bad draft:\n")

    bad_draft = (
        "That's a great question! Let me delve into this crucial topic. "
        "The real estate landscape is shifting dramatically. Furthermore, "
        "it's important to note that yields vary. Here are the key points:\n"
        "1. Location matters\n"
        "2. Timing is everything\n"
        "3. Research is key\n"
        "I hope this helps! Feel free to reach out if you have more questions."
    )

    violations = check_quality(bad_draft, default_rules())
    for v in violations:
        print(f"  [{v.severity}] {v.rule}: matched '{v.matched}'")

    print(f"\n{len(violations)} violations found in {len(bad_draft.split())} words.")


if __name__ == "__main__":
    main()
