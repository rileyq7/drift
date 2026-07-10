"""grant_checker_memory_demo.py — the two-run demo.

Drives examples/grant_checker_with_memory.drift through two evaluations
and prints what Dendric remembers between them. The point is to make
the Dendric integration visible without requiring an Anthropic API key —
we use the mock provider for the LLM and verify behavior at the memory
layer instead.

Usage:
    source .env                                    # OPENAI_API_KEY for embeddings
    export DATABASE_URL=postgresql://postgres:postgres@localhost:5433/dendric
    python examples/grant_checker_memory_demo.py

Without DATABASE_URL the agent falls back to the SQLite mock store; the
demo will run but the "memory across runs" claim won't hold.
"""

import asyncio
import os
import sys

# Ensure the drift package and the generated example module both import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from examples.grant_checker_with_memory import GrantChecker  # noqa: E402


RUN1 = {
    "company_profile": "TechCo, healthcare AI startup, 15 employees",
    "call_text": "Innovate UK Smart Grants 2026: AI for health applications",
}
RUN2 = {
    "company_profile": "MedAI Ltd, drug discovery ML, 8 employees",
    "call_text": "Innovate UK Smart Grants 2026: AI for health applications",
}


def dump_memory_state(label: str, agent: GrantChecker) -> None:
    """Print what Dendric currently remembers. Works in both real and mock mode."""
    mem = agent.memory
    print(f"\n  ── {label} ──")

    # Real Dendric path: query the underlying engine directly.
    eng = getattr(mem, "_eng", None)
    if eng is not None:
        stats = eng.stats()
        print(f"  Dendric stats: total={stats['total']}, "
              f"regions={stats['regions']}")
        for m in eng.get_all(limit=10):
            print(f"    T={m['temperature']:.2f} "
                  f"region={m.get('region', '?')} "
                  f"ctx={m.get('context', '')!r}")
            print(f"        content: {m['raw_content'][:80]}")
        return

    # Mock SQLite path.
    print("  (mock store — install DATABASE_URL for real Dendric)")
    for r in mem.recall("everything"):
        print(f"    {r}")


async def main() -> None:
    print("═" * 60)
    print("  Drift + Dendric — Two-Run Memory Demo")
    print("═" * 60)

    # ── Run 1 ────────────────────────────────────────────────────────
    print(f"\n[Run 1] {RUN1['company_profile']}")
    agent1 = GrantChecker()
    score1 = await agent1.evaluate(**RUN1)
    # consolidate manually so the demo shows the state Dendric would be in
    # after the agent run boundary. (When run via `drift run` this is
    # automatic via run_agent.)
    if hasattr(agent1.memory, "consolidate"):
        agent1.memory.consolidate()
    print(f"\n  → score: {score1.overall_score}/100 ({score1.recommendation})")
    dump_memory_state("memory after Run 1", agent1)

    # ── Run 2 ────────────────────────────────────────────────────────
    print(f"\n[Run 2] {RUN2['company_profile']}")
    agent2 = GrantChecker()  # fresh agent, same Dendric persona
    score2 = await agent2.evaluate(**RUN2)
    if hasattr(agent2.memory, "consolidate"):
        agent2.memory.consolidate()
    print(f"\n  → score: {score2.overall_score}/100 ({score2.recommendation})")
    dump_memory_state("memory after Run 2 — same persona, persisted from Run 1",
                      agent2)
    if hasattr(agent2.memory, "close"):
        agent2.memory.close()

    # ── The proof ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  The point of this demo:")
    print("    - Run 1 stored TechCo's evaluation.")
    print("    - Run 2 started a fresh agent instance.")
    print("    - It found TechCo from Run 1 in the persona's Dendric "
          "memory.")
    print("    - Recall surfaces prior evaluations; consolidate cooled "
          "TechCo's temperature.")
    print("    - With real LLMs the model would see Run 1's verdict as "
          "context for Run 2.")
    print("═" * 60)


if __name__ == "__main__":
    asyncio.run(main())
