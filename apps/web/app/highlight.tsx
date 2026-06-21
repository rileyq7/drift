// Tiny Drift syntax highlighter. Pure SSR — no client JS.

const KEYWORDS = new Set([
  "agent", "schema", "config", "step", "state", "memory", "model", "tool",
  "pipeline", "import", "from", "as", "define", "verb", "if", "let", "return",
  "respond", "for", "each", "in", "parallel", "cached", "manual", "silent",
  "attempt", "recover", "retry", "fail", "on", "remember", "recall", "deja_vu",
  "forget", "tagged", "older", "than", "where", "match", "confident", "prefer",
  "fallback", "never", "upgrade", "when", "stream", "then", "budget", "per",
  "run", "python", "mcp", "rest", "dendric", "true", "false", "null", "is",
  "otherwise", "between", "and", "or", "one", "of", "minimum", "quality",
  "classify", "extract", "summarize", "rate", "generate", "rewrite",
  "answer", "compare", "decide", "translate", "against", "using",
  "considering", "with", "to",
]);

const TYPES = new Set([
  "string", "int", "float", "bool", "number", "list", "map", "optional", "any",
]);

type Tok = { kind: string; text: string };

function tokenize(src: string): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  const n = src.length;
  while (i < n) {
    const ch = src[i];
    // Line comments
    if (ch === "-" && src[i + 1] === "-") {
      let j = i;
      while (j < n && src[j] !== "\n") j++;
      out.push({ kind: "com", text: src.slice(i, j) });
      i = j;
      continue;
    }
    // Strings
    if (ch === '"') {
      let j = i + 1;
      while (j < n && src[j] !== '"') {
        if (src[j] === "\\") j++;
        j++;
      }
      j = Math.min(j + 1, n);
      out.push({ kind: "str", text: src.slice(i, j) });
      i = j;
      continue;
    }
    // Currency
    if (
      (ch === "$" || ch === "£" || ch === "€" || ch === "¥") &&
      /\d/.test(src[i + 1] ?? "")
    ) {
      let j = i + 1;
      while (j < n && /[\d.]/.test(src[j])) j++;
      out.push({ kind: "num", text: src.slice(i, j) });
      i = j;
      continue;
    }
    // Numbers
    if (/\d/.test(ch)) {
      let j = i;
      while (j < n && /[\d.]/.test(src[j])) j++;
      // Optional duration suffix
      if (j < n && /[a-z]/.test(src[j])) {
        let k = j;
        while (k < n && /[a-z]/.test(src[k])) k++;
        const suf = src.slice(j, k);
        if (["ms", "s", "m", "h", "d"].includes(suf)) j = k;
      }
      out.push({ kind: "num", text: src.slice(i, j) });
      i = j;
      continue;
    }
    // Identifiers
    if (/[A-Za-z_]/.test(ch)) {
      let j = i;
      while (j < n && /[A-Za-z0-9_]/.test(src[j])) j++;
      const word = src.slice(i, j);
      let kind = "ident";
      if (KEYWORDS.has(word)) kind = "kw";
      else if (TYPES.has(word)) kind = "typ";
      else if (/^[A-Z]/.test(word)) kind = "typ";
      out.push({ kind, text: word });
      i = j;
      continue;
    }
    // Everything else (punctuation, whitespace) — pass through
    out.push({ kind: "raw", text: ch });
    i++;
  }
  return out;
}

const CLASS: Record<string, string> = {
  kw: "text-[var(--color-syn-kw)] font-medium",
  str: "text-[var(--color-syn-str)]",
  typ: "text-[var(--color-syn-typ)]",
  com: "text-[var(--color-syn-com)] italic",
  num: "text-[var(--color-syn-num)]",
  ident: "text-[var(--color-ink)]",
  raw: "text-[var(--color-ink-soft)]",
};

export function Drift({ code }: { code: string }) {
  const toks = tokenize(code.trim());
  return (
    <pre className="font-mono text-[13.5px] leading-[1.65] whitespace-pre overflow-x-auto rounded-xl border border-[var(--color-cream-deep)] bg-[var(--color-cream-soft)] p-5 md:p-6 text-[var(--color-ink-soft)]">
      <code>
        {toks.map((t, i) => (
          <span key={i} className={CLASS[t.kind]}>
            {t.text}
          </span>
        ))}
      </code>
    </pre>
  );
}

export function Shell({ children }: { children: string }) {
  return (
    <div className="font-mono text-[13.5px] rounded-xl border border-[var(--color-cream-deep)] bg-[var(--color-cream-soft)] p-5 md:p-6 text-[var(--color-ink-soft)] whitespace-pre overflow-x-auto">
      <span className="text-[var(--color-sage)]">$ </span>
      <span className="text-[var(--color-ink)]">{children}</span>
    </div>
  );
}
