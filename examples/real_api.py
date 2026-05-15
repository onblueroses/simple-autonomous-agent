"""Run the full agent pipeline with real LLM calls via OpenRouter.

Setup:
    1. Get a free API key at https://openrouter.ai/keys (no credit card needed)
    2. Set it: export OPENROUTER_API_KEY="<your-openrouter-key>"
    3. Run: python examples/real_api.py

Uses three free models for three jobs. Verified 2026-05-15 against
https://openrouter.ai/api/v1/models.
"""

import os
import re
import sys
import urllib.request
import urllib.parse

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


SCORER_MODEL = "nvidia/nemotron-nano-9b-v2:free"  # verified: 2026-05-15
REASONER_MODEL = "deepseek/deepseek-v4-flash:free"  # verified: 2026-05-15
WRITER_MODEL = "google/gemma-4-31b-it:free"  # verified: 2026-05-15


# --- Grounding: DuckDuckGo search (no API key, stdlib only) ---


def ddg_search(query: str) -> str:
    """DEMO ONLY. Scrapes DDG's HTML page and regex-parses snippets.

    The HTML structure is undocumented and changes without notice; expect this
    to break. For production grounding, plug in Tavily, SerpAPI, or Brave Search.
    See the commented Tavily block below as a starting template.
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
    persona = load_persona(
        os.path.join(os.path.dirname(__file__), "..", "personas", "analyst.yaml")
    )

    config = PipelineConfig(
        scorer=ModelConfig(SCORER_MODEL, max_tokens=256),
        reasoner=ModelConfig(REASONER_MODEL, max_tokens=1024),
        writer=ModelConfig(WRITER_MODEL, max_tokens=1024),
        scorer_client=client,
        writer_client=client,
        ground_fn=ddg_search,
        quality_rules=default_rules(),
    )

    item = {
        "id": "demo-1",
        "text": "How do open-source dependency audit tools compare on signal-to-noise for medium-sized Python codebases?",
    }

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
        print("\n--- Errors ---")
        for e in result.errors:
            print(f"  {e}")

    # --- Quality rules demo ---
    print("\n\n=== Quality Rules Demo ===")
    print("Running quality checks on a deliberately bad draft:\n")

    bad_draft = (
        "That's a great question! Let me delve into this crucial topic. "
        "The dependency-audit landscape is shifting dramatically. Furthermore, "
        "it's important to note that signal-to-noise varies. Here are the key points:\n"
        "1. Coverage matters\n"
        "2. Speed matters\n"
        "3. False positives matter\n"
        "I hope this helps! Feel free to reach out if you have more questions."
    )

    violations = check_quality(bad_draft, default_rules())
    for v in violations:
        print(f"  [{v.severity}] {v.rule}: matched '{v.matched}'")

    print(f"\n{len(violations)} violations found in {len(bad_draft.split())} words.")


if __name__ == "__main__":
    main()
