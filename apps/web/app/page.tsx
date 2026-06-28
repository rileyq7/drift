import { Drift, Shell } from "./highlight";
import { TerminalDemo } from "./TerminalDemo";

const SAMPLE = `agent InboxTriage {
  model: "gpt-5.4-nano"
  budget: $0.10 per run

  step triage(emails: list<string>) -> list<EmailAnalysis> {
    let analyses = []
    for each email in emails parallel {
      let analysis = classify email as EmailAnalysis
      analyses.add(analysis)
    }
    for each a in analyses {
      if a.priority == "urgent" {
        respond "URGENT — {a.subject}: {a.summary}"
      }
    }
    return analyses
  }
}`;

const CONFIDENT = `step assess(input: string) -> Decision {
  let scored = rate input against rubric as confident<Decision>
  if scored is confident {
    return scored.value
  }
  fail "low confidence — escalate"
}`;

const RECOVER = `attempt {
  let data = fetch_url(url)
  return summarize data as string
} recover from {
  RateLimited    -> retry
  BudgetExceeded -> fail "out of budget"
  any error      -> respond "fallback"
}`;

export default function Page() {
  return (
    <main className="min-h-screen">
      <div className="mx-auto max-w-3xl px-6 pb-32 pt-20 md:pt-28">
        {/* Hero */}
        <header className="mb-16">
          <div className="mb-8 inline-flex items-center gap-2 rounded-full border border-[var(--color-cream-deep)] bg-[var(--color-cream-soft)] px-3 py-1 text-xs text-[var(--color-ink-muted)]">
            <span className="size-1.5 rounded-full bg-[var(--color-sage)]" />
            v0.1.1 · MIT · 352 tests passing
          </div>
          <h1 className="text-[clamp(48px,8vw,80px)] font-semibold leading-[0.95] tracking-[-0.03em] text-[var(--color-sage-dark)]">
            drift
          </h1>
          <p className="mt-6 max-w-xl text-[19px] leading-relaxed text-[var(--color-ink-soft)]">
            An intent-based language for LLM agents.
            <br />
            Write agents in English. Transpile to async Python.
          </p>

          <div className="mt-10 flex flex-wrap items-center gap-3">
            <a
              href="https://github.com/rileyq7/drift"
              className="inline-flex items-center gap-2 rounded-lg bg-[var(--color-sage-dark)] px-4 py-2.5 text-sm font-medium text-[var(--color-cream)] transition hover:bg-[var(--color-sage-deep)]"
            >
              GitHub →
            </a>
            <a
              href="https://github.com/rileyq7/drift/blob/main/LLM.md"
              className="inline-flex items-center gap-2 rounded-lg border border-[var(--color-cream-deep)] bg-[var(--color-cream-soft)] px-4 py-2.5 text-sm text-[var(--color-ink)] transition hover:border-[var(--color-sage-soft)]"
            >
              For coding agents
            </a>
            <a
              href="https://github.com/rileyq7/drift/blob/main/docs/cookbook.md"
              className="inline-flex items-center gap-2 rounded-lg border border-[var(--color-cream-deep)] bg-[var(--color-cream-soft)] px-4 py-2.5 text-sm text-[var(--color-ink)] transition hover:border-[var(--color-sage-soft)]"
            >
              Cookbook
            </a>
          </div>

          <div className="mt-6 max-w-md">
            <Shell>pip install drift-lang</Shell>
          </div>
        </header>

        {/* Live terminal demo */}
        <section className="mb-20">
          <h2 className="mb-1 text-xs font-medium uppercase tracking-[0.18em] text-[var(--color-sage-deep)]">
            Memory that actually works across runs
          </h2>
          <p className="mb-5 text-sm text-[var(--color-ink-muted)]">
            Two fresh agent instances. One persistent Dendric memory. Run 2&apos;s LLM
            reasoning visibly cites Run 1 — by name, with numerical comparisons.
            Real <code className="font-mono text-[12.5px]">gpt-5.4</code>, real output.
          </p>
          <TerminalDemo />
        </section>

        {/* Code sample */}
        <section className="mb-20">
          <h2 className="mb-1 text-xs font-medium uppercase tracking-[0.18em] text-[var(--color-sage-deep)]">
            What a Drift program looks like
          </h2>
          <p className="mb-5 text-sm text-[var(--color-ink-muted)]">
            A full agent — model choice, budget, parallel fan-out, structured
            classification, conditional output.
          </p>
          <Drift code={SAMPLE} />
          <p className="mt-4 text-sm leading-relaxed text-[var(--color-ink-muted)]">
            Live against OpenAI:{" "}
            <span className="text-[var(--color-ink)]">5 emails</span>,{" "}
            <span className="text-[var(--color-ink)]">1.82s</span>,{" "}
            <span className="rounded bg-[var(--color-sage-soft)]/30 px-1.5 py-0.5 font-mono text-xs text-[var(--color-sage-dark)]">
              $0.0092
            </span>
            , returned 5 typed dataclasses.
          </p>
        </section>

        {/* Why */}
        <section className="mb-20">
          <h2 className="mb-4 text-xs font-medium uppercase tracking-[0.18em] text-[var(--color-sage-deep)]">
            Why a language
          </h2>
          <div className="space-y-4 text-[15.5px] leading-relaxed text-[var(--color-ink-soft)]">
            <p>
              Agent code is mostly orchestration: pick a model, call it with
              structure, branch on the result, escalate when uncertain, retry
              on rate limits, track costs. That orchestration reads worse than
              the prose that describes it.
            </p>
            <p>
              Drift makes the prose runnable. The transpiler emits async Python
              that runs on a thin runtime handling provider routing, budget
              caps, schema validation, retries, memory, and MCP tool calls. You
              write the spec; the runtime handles the plumbing.
            </p>
          </div>
        </section>

        {/* Features */}
        <section className="mb-20">
          <h2 className="mb-6 text-xs font-medium uppercase tracking-[0.18em] text-[var(--color-sage-deep)]">
            What's in the box
          </h2>
          <ul className="grid gap-x-8 gap-y-4 md:grid-cols-2">
            {[
              [
                "Multi-provider routing",
                "prefer / fallback / upgrade. OpenAI + Anthropic, MCP tools, REST inline.",
              ],
              [
                "Strict JSON output",
                "Schemas become provider-side JSON Schema. Almost-JSON failures: gone.",
              ],
              [
                "Confidence gating",
                "confident<T> wrapper or explicit confidence field. Branch on certainty.",
              ],
              [
                "Parallel by default",
                "for each x in xs parallel — asyncio.gather underneath.",
              ],
              [
                "Cost tracking",
                "Every run prints a cost report. Hard budget caps.",
              ],
              [
                "Memory",
                "Dendric (semantic) or SQLite. remember / recall / deja_vu / forget.",
              ],
              [
                "Structured errors",
                "recover from { BudgetExceeded -> ..., RateLimited -> retry }",
              ],
              [
                "Dev tools",
                "drift run --watch, drift fmt, drift check, VS Code extension, source-mapped errors.",
              ],
            ].map(([title, desc]) => (
              <li key={title} className="border-l-2 border-[var(--color-sage-soft)] pl-4">
                <div className="text-sm font-medium text-[var(--color-ink)]">{title}</div>
                <div className="mt-1 text-[14px] leading-snug text-[var(--color-ink-muted)]">
                  {desc}
                </div>
              </li>
            ))}
          </ul>
        </section>

        {/* Confidence gating */}
        <section className="mb-20">
          <h2 className="mb-2 text-xs font-medium uppercase tracking-[0.18em] text-[var(--color-sage-deep)]">
            Confidence-gated branching
          </h2>
          <p className="mb-5 text-sm text-[var(--color-ink-muted)]">
            Cheap model when sure. Escalate when not.
          </p>
          <Drift code={CONFIDENT} />
        </section>

        {/* Error recovery */}
        <section className="mb-20">
          <h2 className="mb-2 text-xs font-medium uppercase tracking-[0.18em] text-[var(--color-sage-deep)]">
            Structured error handling
          </h2>
          <p className="mb-5 text-sm text-[var(--color-ink-muted)]">
            Typed recover arms. retry / fail / fallback in one block.
          </p>
          <Drift code={RECOVER} />
        </section>

        {/* Quickstart */}
        <section className="mb-20">
          <h2 className="mb-2 text-xs font-medium uppercase tracking-[0.18em] text-[var(--color-sage-deep)]">
            30 seconds to your first agent
          </h2>
          <p className="mb-5 text-sm text-[var(--color-ink-muted)]">
            No API key required — falls back to a mock provider that returns
            deterministic stub data. Drop a key in <code className="font-mono text-[12.5px]">.env</code> for real calls.
          </p>
          <div className="space-y-3">
            <Shell>pip install drift-lang</Shell>
            <Shell>drift new hello && cd hello</Shell>
            <Shell>{`drift run hello.drift --input '{"name":"Riley"}'`}</Shell>
          </div>
        </section>

        {/* For agents */}
        <section className="mb-20">
          <h2 className="mb-3 text-xs font-medium uppercase tracking-[0.18em] text-[var(--color-sage-deep)]">
            For coding agents
          </h2>
          <p className="text-[15.5px] leading-relaxed text-[var(--color-ink-soft)]">
            If you're a coding agent (Claude, Cursor, Copilot) writing Drift on
            a user's behalf: load{" "}
            <a
              href="https://github.com/rileyq7/drift/blob/main/LLM.md"
              className="text-[var(--color-sage-deep)] underline decoration-[var(--color-sage-soft)] underline-offset-4 hover:decoration-[var(--color-sage-deep)]"
            >
              LLM.md
            </a>{" "}
            first. It's a single-file complete reference, cross-checked against
            the parser. Cold-start agents have written working programs from it
            on first try.
          </p>
        </section>

        {/* Comparison */}
        <section className="mb-20">
          <h2 className="mb-6 text-xs font-medium uppercase tracking-[0.18em] text-[var(--color-sage-deep)]">
            Why not LangGraph / CrewAI / framework X?
          </h2>
          <div className="overflow-hidden rounded-xl border border-[var(--color-cream-deep)]">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-[var(--color-cream-soft)] text-left text-xs uppercase tracking-wider text-[var(--color-ink-muted)]">
                  <th className="px-5 py-3"></th>
                  <th className="px-5 py-3 font-medium">Drift</th>
                  <th className="px-5 py-3 font-medium">Python frameworks</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--color-cream-deep)]">
                {[
                  ["Code shape", "Declarative blocks", "Imperative classes / graphs"],
                  ["Budget caps", "Built into syntax", "Wrapper or none"],
                  ["Provider routing", "One line", "Manual or via wrapper"],
                  ["Memory", "One line + Dendric", "Bring your own"],
                  ["Schema enforcement", "Strict via provider", "Manual JSON validation"],
                  ["Best when", "Spec reads better than glue", "You need full Python control"],
                ].map(([label, a, b]) => (
                  <tr key={label}>
                    <td className="px-5 py-3 font-medium text-[var(--color-ink)]">{label}</td>
                    <td className="px-5 py-3 text-[var(--color-ink-soft)]">{a}</td>
                    <td className="px-5 py-3 text-[var(--color-ink-muted)]">{b}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-4 text-sm text-[var(--color-ink-muted)]">
            Drift isn&apos;t a general-purpose language. It&apos;s orchestration
            sugar. When your agent IS orchestration, it wins. When you need
            fine-grained control flow, drop to Python.
          </p>
        </section>

        {/* Footer */}
        <footer className="border-t border-[var(--color-cream-deep)] pt-8 text-sm">
          <div className="flex flex-wrap items-center justify-between gap-3 text-[var(--color-ink-muted)]">
            <div>MIT · 352 tests · 0.1.1 alpha</div>
            <a
              href="https://github.com/rileyq7/drift"
              className="hover:text-[var(--color-sage-deep)]"
            >
              github.com/rileyq7/drift
            </a>
          </div>
        </footer>
      </div>
    </main>
  );
}
