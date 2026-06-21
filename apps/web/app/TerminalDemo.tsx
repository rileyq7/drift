"use client";

import { useEffect, useRef, useState } from "react";

type Step =
  | { t: "prompt"; v: string; d?: number }
  | { t: "type"; v: string; s?: number }
  | { t: "line"; v: string; d?: number }
  | { t: "pause"; ms: number }
  | { t: "cursor" };

const SCRIPT: Step[] = [
  { t: "prompt", v: "$ " },
  { t: "type", v: "pip install drift-lang", s: 28 },
  { t: "pause", ms: 400 },
  { t: "line", v: `<span class="dim">Successfully installed drift-lang-0.1.1</span>`, d: 600 },
  { t: "line", v: "" },

  // ── Run 1 invocation ──
  { t: "prompt", v: "$ " },
  { t: "type", v: `drift run grant_checker.drift --input '{"company":"TechCo, 15 emp, 18mo"}'`, s: 22 },
  { t: "pause", ms: 700 },
  { t: "line", v: "" },
  { t: "line", v: `<span class="dim">  ▸ provider: openai · loaded 1 var from .env</span>`, d: 30 },
  { t: "line", v: `<span class="dim">  ✓ Transpiled → grant_checker.py</span>`, d: 50 },
  { t: "line", v: "" },
  { t: "line", v: `<span class="sep">══════════════════════════════════════════════════════════════════════</span>`, d: 20 },
  { t: "line", v: `<span class="head">  Drift — Running GrantChecker</span>`, d: 20 },
  { t: "line", v: `<span class="sep">══════════════════════════════════════════════════════════════════════</span>`, d: 20 },
  { t: "line", v: "" },

  // Run 1
  { t: "line", v: `<span class="label">[Run 1]</span> TechCo Ltd — UK-based healthcare AI startup, 15 employees,`, d: 150 },
  { t: "line", v: `        Series A, clinical diagnosis support using deep learning over`, d: 30 },
  { t: "line", v: `        EHR data, 18 months into development.`, d: 30 },
  { t: "line", v: `  <span class="dim">▸</span>  compare_score() via <span class="val">gpt-5.4</span>`, d: 400 },
  { t: "pause", ms: 1800 },
  { t: "line", v: `  Run 1 <span class="arrow">→</span> score: <span class="score">72.0/100</span> (confidence 0.63)`, d: 40 },
  { t: "line", v: "" },
  { t: "line", v: `  <span class="dim">Run 1 reasoning (selected):</span>`, d: 200 },
  { t: "line", v: `    <span class="reason">"Compared with the prior evaluations in the provided context,</span>`, d: 30 },
  { t: "line", v: `    <span class="reason"> there are no named prior companies available to benchmark against."</span>`, d: 30 },
  { t: "line", v: `    <span class="note">← memory was truly empty going into Run 1</span>`, d: 60 },
  { t: "line", v: "" },
  { t: "pause", ms: 1600 },

  // ── Run 2 invocation (separate process, same persona memory) ──
  { t: "prompt", v: "$ " },
  { t: "type", v: `drift run grant_checker.drift --input '{"company":"MedAI, 12 emp, 20mo"}'`, s: 22 },
  { t: "pause", ms: 700 },
  { t: "line", v: "" },
  { t: "line", v: `<span class="note">  ▸ same persona key — recalling prior evaluations from Dendric</span>`, d: 60 },
  { t: "line", v: "" },

  // Run 2 — the citation moment
  { t: "line", v: `<span class="label">[Run 2]</span> MedAI Ltd — UK-based healthcare AI startup, 12 employees,`, d: 150 },
  { t: "line", v: `        Series A, clinical decision support using deep learning over`, d: 30 },
  { t: "line", v: `        patient records, 20 months into development.`, d: 30 },
  { t: "line", v: `  <span class="dim">▸</span>  compare_score() via <span class="val">gpt-5.4</span>`, d: 400 },
  { t: "pause", ms: 1800 },
  { t: "line", v: "" },
  { t: "line", v: `  Run 2 <span class="arrow">→</span> score: <span class="score">70.0/100</span> (confidence 0.72)`, d: 40 },
  { t: "line", v: `  <span class="dim">Run 2 reasoning (selected):</span>`, d: 300 },
  { t: "line", v: `    <span class="reason">"The two key gaps are the same ones that limited</span> <span class="cite">TechCo Ltd</span>`, d: 50 },
  { t: "line", v: `    <span class="reason"> in the prior evaluation: technical novelty and commercialization</span>`, d: 40 },
  { t: "line", v: `    <span class="reason"> pathway."</span>`, d: 40 },
  { t: "line", v: "" },
  { t: "line", v: `    <span class="reason">"Compared with</span> <span class="cite">TechCo Ltd</span><span class="reason">, MedAI Ltd is very similar in sector,</span>`, d: 50 },
  { t: "line", v: `    <span class="reason"> modality, and grant fit, and I would assess it as slightly</span>`, d: 40 },
  { t: "line", v: `    <span class="reason"> weaker overall. MedAI Ltd has a slightly smaller team</span> <span class="emph">(12 vs 15)</span><span class="reason">,</span>`, d: 40 },
  { t: "line", v: `    <span class="reason"> and while it is a bit further into development</span> <span class="emph">(20 vs 18)</span><span class="reason">,</span>`, d: 40 },
  { t: "line", v: `    <span class="reason"> that does not materially improve the case because the same core</span>`, d: 40 },
  { t: "line", v: `    <span class="reason"> evidence gaps remain unresolved."</span>`, d: 40 },
  { t: "line", v: "" },
  { t: "line", v: `    <span class="note">← Run 2 cited TechCo by name 3× and made side-by-side</span>`, d: 50 },
  { t: "line", v: `    <span class="note">   numerical comparisons (12 vs 15, 20 vs 18) that are only</span>`, d: 50 },
  { t: "line", v: `    <span class="note">   possible if Run 1&apos;s stored data reached Run 2&apos;s prompt.</span>`, d: 50 },
  { t: "line", v: "" },

  { t: "line", v: `<span class="sep">══════════════════════════════════════════════════════════════════════</span>`, d: 20 },
  { t: "line", v: `  <span class="dim">Two fresh agent instances. One persistent memory. That&apos;s Drift.</span>`, d: 80 },
  { t: "line", v: `  <span class="dim">pip install drift-lang · github.com/rileyq7/drift</span>`, d: 80 },
  { t: "line", v: `<span class="sep">══════════════════════════════════════════════════════════════════════</span>`, d: 20 },
  { t: "line", v: "" },
  { t: "prompt", v: "$ ", d: 400 },
  { t: "cursor" },
];

export function TerminalDemo() {
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const abortRef = useRef<{ aborted: boolean }>({ aborted: false });
  const [, setNonce] = useState(0);

  useEffect(() => {
    abortRef.current = { aborted: false };
    void play(bodyRef.current!, abortRef.current);
    return () => {
      abortRef.current.aborted = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const restart = () => {
    abortRef.current.aborted = true;
    if (bodyRef.current) bodyRef.current.innerHTML = "";
    abortRef.current = { aborted: false };
    setNonce((n) => n + 1);
    setTimeout(() => void play(bodyRef.current!, abortRef.current), 50);
  };

  const skip = () => {
    abortRef.current.aborted = true;
    if (!bodyRef.current) return;
    bodyRef.current.innerHTML = "";
    renderAll(bodyRef.current);
  };

  return (
    <div className="mt-12">
      <style>{TERMINAL_CSS}</style>
      <div className="terminal-wrap">
        <div className="terminal">
          <div className="titlebar">
            <div className="dot dot-red" />
            <div className="dot dot-yellow" />
            <div className="dot dot-green" />
            <div className="titlebar-text">drift — memory demo</div>
          </div>
          <div className="terminal-body" ref={bodyRef} />
        </div>
        <div className="controls">
          <button onClick={restart}>↻ Replay</button>
          <button onClick={skip}>Skip →</button>
        </div>
      </div>
    </div>
  );
}

function scrollTo(body: HTMLElement) {
  body.scrollTop = body.scrollHeight;
}

function sleep(ms: number, signal: { aborted: boolean }) {
  return new Promise<void>((resolve) => {
    const start = Date.now();
    const tick = () => {
      if (signal.aborted || Date.now() - start >= ms) return resolve();
      setTimeout(tick, 16);
    };
    tick();
  });
}

async function play(body: HTMLElement, signal: { aborted: boolean }) {
  for (const s of SCRIPT) {
    if (signal.aborted) return;
    if (s.t === "prompt") {
      const el = document.createElement("div");
      el.className = "line";
      el.innerHTML = `<span class="prompt">${s.v}</span>`;
      body.appendChild(el);
      scrollTo(body);
      await sleep(s.d ?? 500, signal);
    } else if (s.t === "type") {
      const parent = body.lastElementChild;
      if (!parent) continue;
      const span = document.createElement("span");
      span.className = "cmd";
      parent.appendChild(span);
      for (const ch of s.v) {
        if (signal.aborted) {
          span.textContent = s.v;
          break;
        }
        span.textContent += ch;
        scrollTo(body);
        await sleep(s.s ?? 35, signal);
      }
    } else if (s.t === "line") {
      const el = document.createElement("div");
      el.className = "line";
      el.innerHTML = s.v;
      body.appendChild(el);
      scrollTo(body);
      await sleep(s.d ?? 40, signal);
    } else if (s.t === "pause") {
      await sleep(s.ms, signal);
    } else if (s.t === "cursor") {
      const p = body.lastElementChild ?? body;
      const c = document.createElement("span");
      c.className = "cursor";
      p.appendChild(c);
    }
  }
}

function renderAll(body: HTMLElement) {
  for (const s of SCRIPT) {
    if (s.t === "prompt") {
      const el = document.createElement("div");
      el.className = "line";
      el.innerHTML = `<span class="prompt">${s.v}</span>`;
      body.appendChild(el);
    } else if (s.t === "type") {
      const parent = body.lastElementChild;
      if (!parent) continue;
      const span = document.createElement("span");
      span.className = "cmd";
      span.textContent = s.v;
      parent.appendChild(span);
    } else if (s.t === "line") {
      const el = document.createElement("div");
      el.className = "line";
      el.innerHTML = s.v;
      body.appendChild(el);
    } else if (s.t === "cursor") {
      const c = document.createElement("span");
      c.className = "cursor";
      (body.lastElementChild ?? body).appendChild(c);
    }
  }
  scrollTo(body);
}

const TERMINAL_CSS = `
.terminal-wrap {
  width: 100%;
  max-width: 820px;
  margin: 0 auto;
}
.terminal {
  background: #faf5e7;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid var(--color-cream-deep);
  box-shadow:
    0 1px 0 rgba(255,255,255,0.6) inset,
    0 12px 28px rgba(63, 74, 54, 0.08),
    0 2px 6px rgba(63, 74, 54, 0.05);
}
.titlebar {
  background: var(--color-cream-soft);
  padding: 11px 14px;
  display: flex;
  align-items: center;
  gap: 7px;
  border-bottom: 1px solid var(--color-cream-deep);
}
.dot {
  width: 11px;
  height: 11px;
  border-radius: 50%;
  border: 1px solid rgba(0,0,0,0.06);
}
.dot-red { background: #e88b7d; }
.dot-yellow { background: #e7c275; }
.dot-green { background: #a4b196; }
.titlebar-text {
  flex: 1;
  text-align: center;
  color: var(--color-ink-muted);
  font-size: 11.5px;
  letter-spacing: 0.5px;
  font-family: var(--font-mono);
}
.terminal-body {
  padding: 20px 24px 24px;
  font-family: var(--font-mono);
  font-size: 12.5px;
  line-height: 1.7;
  color: var(--color-ink-soft);
  min-height: 480px;
  max-height: 70vh;
  overflow-y: auto;
}
.terminal-body::-webkit-scrollbar { width: 6px; }
.terminal-body::-webkit-scrollbar-track { background: transparent; }
.terminal-body::-webkit-scrollbar-thumb { background: var(--color-cream-deep); border-radius: 3px; }
.terminal-body .line {
  white-space: pre-wrap;
  word-break: break-word;
}
.terminal-body .prompt { color: var(--color-sage-deep); }
.terminal-body .cmd { color: var(--color-ink); font-weight: 600; }
.terminal-body .head { color: var(--color-sage-dark); font-weight: 700; }
.terminal-body .label { color: var(--color-sage-deep); font-weight: 700; }
.terminal-body .val { color: var(--color-syn-typ); }
.terminal-body .score { color: var(--color-syn-typ); font-weight: 700; }
.terminal-body .dim { color: var(--color-ink-muted); }
.terminal-body .reason { color: var(--color-ink-soft); }
.terminal-body .cite {
  color: var(--color-sage-dark);
  font-weight: 700;
  background: rgba(164, 177, 150, 0.32);
  padding: 0 4px;
  border-radius: 3px;
}
.terminal-body .arrow { color: var(--color-sage); }
.terminal-body .note { color: var(--color-sage-deep); font-style: italic; }
.terminal-body .sep { color: var(--color-cream-deep); }
.terminal-body .emph {
  color: var(--color-syn-typ);
  font-weight: 700;
}
.terminal-body .cursor {
  display: inline-block;
  width: 7px;
  height: 14px;
  background: var(--color-sage-deep);
  vertical-align: text-bottom;
  margin-left: 1px;
  animation: drift-blink 1s step-end infinite;
}
@keyframes drift-blink { 50% { opacity: 0; } }
.controls {
  display: flex;
  justify-content: center;
  gap: 12px;
  margin-top: 14px;
}
.controls button {
  font-family: var(--font-mono);
  font-size: 12px;
  padding: 7px 18px;
  border-radius: 6px;
  border: 1px solid var(--color-cream-deep);
  background: var(--color-cream-soft);
  color: var(--color-ink-soft);
  cursor: pointer;
  transition: all 0.15s;
}
.controls button:hover {
  background: var(--color-cream);
  color: var(--color-ink);
  border-color: var(--color-sage-soft);
}
`;
