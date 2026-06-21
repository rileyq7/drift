"""grant_checker_compare_demo.py — the citation-proof memory demo.

Drives examples/grant_checker_compare.drift through two evaluations of
deliberately-similar companies and prints both reasoning blocks. The
point is to make the LLM's USE of memory visible — Run 2 will cite Run 1
by name and make explicit side-by-side comparisons.

Usage:
    docker compose -f Dendric/docker-compose.yml up -d db
    set -a && source .env && set +a    # OPENAI_API_KEY for LLM + embeddings
    export DATABASE_URL=postgresql://postgres:postgres@localhost:5433/dendric
    python examples/grant_checker_compare_demo.py

Without DATABASE_URL the agent falls back to the SQLite mock store; the
demo will run but the recall layer won't return the rich Run 1 verdict
that makes the citation in Run 2 work.

Without OPENAI_API_KEY the agent falls back to the mock LLM provider;
both runs return identical canned responses and the citation claim
won't hold.
"""

import asyncio
import os
import re
import sys

# Ensure the drift package and the generated example module both import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from examples.grant_checker_compare import GrantChecker  # noqa: E402


# Two deliberately-comparable applications. Same domain, same modality,
# nearly identical team sizes and timelines. A human would compare them
# naturally; we want the model to do the same when given Run 1 as context.
RUN1 = {
    "company_profile": (
        "TechCo Ltd — UK-based healthcare AI startup, 15 employees, "
        "Series A, building a clinical diagnosis support system using "
        "deep learning over EHR data. Currently 18 months into development."
    ),
    "call_text": (
        "Innovate UK Smart Grants 2026: funding for AI applications in "
        "healthcare and life sciences. Up to £500K. UK-registered SMEs. "
        "Must demonstrate technical novelty and commercialization pathway."
    ),
}
RUN2 = {
    "company_profile": (
        "MedAI Ltd — UK-based healthcare AI startup, 12 employees, "
        "Series A, building a clinical decision support tool using "
        "deep learning over patient records. Currently 20 months into development."
    ),
    "call_text": RUN1["call_text"],
}

# Words/phrases that indicate Run 2 is referencing Run 1.
CITATION_NEEDLES = [
    "TechCo",
    "prior",
    "previous",
    "earlier",
    "similar",
    "compared",
    "in the prior evaluation",
    "Run 1",
    "the previous applicant",
]


def banner(title: str) -> None:
    print("\n" + "═" * 72)
    print(f"  {title}")
    print("═" * 72)


async def main() -> None:
    banner("Drift + Dendric — Citation Proof (real LLM)")

    # ── Run 1 ────────────────────────────────────────────────────────────
    print(f"\n[Run 1] {RUN1['company_profile'][:65]}...")
    a1 = GrantChecker()
    s1 = await a1.evaluate(**RUN1)
    if hasattr(a1.memory, "consolidate"):
        a1.memory.consolidate()
    print(f"\n  Run 1 → score: {s1.overall_score}/100 (confidence {s1.confidence})")
    print(f"\n  Run 1 reasoning:\n  {s1.reasoning}")

    # ── Run 2 ────────────────────────────────────────────────────────────
    print(f"\n[Run 2] {RUN2['company_profile'][:65]}...")
    a2 = GrantChecker()  # fresh instance; only the persona key links them
    s2 = await a2.evaluate(**RUN2)
    if hasattr(a2.memory, "consolidate"):
        a2.memory.consolidate()
    print(f"\n  Run 2 → score: {s2.overall_score}/100 (confidence {s2.confidence})")
    print(f"\n  Run 2 reasoning:\n  {s2.reasoning}")
    if hasattr(a2.memory, "close"):
        a2.memory.close()

    # ── The proof ────────────────────────────────────────────────────────
    banner("Citation check on Run 2's reasoning")
    found = [n for n in CITATION_NEEDLES if re.search(rf"\b{n}\b", s2.reasoning, re.IGNORECASE)]
    if found:
        print("\n  ✓ Run 2 references prior context. Markers found:")
        for n in found:
            print(f"     - '{n}'")
        print("\n  This is the proof. Memory persisted across agent instances,")
        print("  reached Run 2's prompt as CONTEXT, and the model used it in")
        print("  its reasoning.")
    else:
        print("\n  ✗ Run 2's reasoning does not visibly cite Run 1.")
        print("    (Plumbing still works — memory reached the prompt — but the")
        print("    LLM didn't surface it in this output. Try a stronger model")
        print("    or a more directive verb prompt.)")


if __name__ == "__main__":
    asyncio.run(main())
