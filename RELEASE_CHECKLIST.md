# Release Checklist (user-only steps)

Everything that needs your credentials. I can't do these for you — but each one is < 15 minutes once you have the right account/token.

Suggested order: **PyPI → VS Code → landing page DNS → social**. PyPI is the gate; without it `pip install drift-lang` doesn't work for anyone but you.

---

## 1. Publish to PyPI

Goal: `pip install drift-lang` works for the world.

**One-time setup:**

1. Create accounts:
   - https://pypi.org/account/register/
   - https://test.pypi.org/account/register/ (separate account)
2. Enable 2FA on both (PyPI requires it).
3. Generate an API token on PyPI:
   - https://pypi.org/manage/account/token/
   - Scope: "Entire account" for the first upload; tighten later.
   - Save it somewhere; you'll paste it once.

**Build and upload:**

```bash
cd /Users/rileycoleman/drift
.venv/bin/pip install --quiet build twine
.venv/bin/python -m build              # → dist/drift_lang-0.1.0-py3-none-any.whl + .tar.gz
.venv/bin/python -m twine check dist/* # sanity-check metadata

# Optional dry-run to TestPyPI first:
.venv/bin/python -m twine upload --repository testpypi dist/*
#   user: __token__
#   password: <your TestPyPI token>
# then verify:
pip install --index-url https://test.pypi.org/simple/ drift-lang

# Real upload:
.venv/bin/python -m twine upload dist/*
#   user: __token__
#   password: <your PyPI token>
```

After: bump `version` in `pyproject.toml` before every subsequent upload. PyPI rejects re-uploads of the same version.

---

## 2. Publish the VS Code extension

Goal: searchable on the VS Code marketplace.

**One-time setup:**

1. Create publisher at https://marketplace.visualstudio.com/manage
   - Publisher ID: `rileyq7` (must match `package.json`'s `publisher` field)
   - If you pick a different ID, edit `vscode-drift/package.json` and rebuild.
2. Generate Azure DevOps PAT:
   - https://dev.azure.com → User Settings (top right) → Personal Access Tokens
   - Organization: **All accessible organizations**
   - Scope: **Marketplace → Manage**
   - Lifetime: 90 days is the max; renew when it expires.

**Build and publish:**

```bash
cd /Users/rileycoleman/drift/vscode-drift
npx --yes @vscode/vsce login rileyq7      # paste PAT once
npx @vscode/vsce package --no-dependencies  # builds drift-lang-0.1.0.vsix
npx @vscode/vsce publish
```

Verify: search "Drift" at https://marketplace.visualstudio.com/

After: bump `version` in `vscode-drift/package.json` before re-publishing.

---

## 3. Point a domain at the landing page

The static landing page lives at `web/index.html`. Three hosting options:

**Option A — GitHub Pages (free, fastest)**

```bash
cd /Users/rileycoleman/drift
# Create a docs branch or move web/ to a dedicated repo
mkdir -p docs-site && cp web/index.html docs-site/
# Settings → Pages → Source: main branch /docs-site folder
```

Custom domain on Pages: Settings → Pages → Custom domain → `dendric.dev` (or `drift.dev`). Then add a CNAME at your registrar pointing to `rileyq7.github.io`.

**Option B — Vercel (free for personal projects)**

```bash
# from project root
npx vercel
# Follow prompts; tell it the output dir is web/
# Then attach the domain in Vercel dashboard
```

**Option C — Cloudflare Pages**

Same shape — upload `web/`, attach domain.

DNS notes:
- For apex domain (`dendric.dev`): A record → 185.199.108.153 (GitHub Pages IP)
- For `www.dendric.dev`: CNAME → `rileyq7.github.io`

---

## 4. Set the GitHub repo description and topics

Quick polish, big SEO/discovery win.

1. Go to https://github.com/rileyq7/drift
2. Click the gear icon next to **About** in the right sidebar.
3. Description: `An intent-based language for agentic systems. Write agents in English, run them as async Python.`
4. Website: your domain once it's live.
5. Topics: `ai-agents`, `llm`, `transpiler`, `dsl`, `python`, `claude`, `gpt`, `openai`, `anthropic`, `mcp`, `agentic-ai`

Pin the repo on your profile.

---

## 5. Social launch

Post-PyPI/marketplace. Below is copy you can lift verbatim; edit to taste.

### Tweet/X (under 280 chars)

> just shipped drift — an intent-based language for LLM agents.
>
> write `agent X { step y(...) { classify foo as Bar } }`
> transpiles to async python with budgets, retries, memory, mcp tools, source maps.
>
> pip install drift-lang
>
> github.com/rileyq7/drift

### HN "Show HN" title

> Show HN: Drift – Write LLM agents as declarative blocks; transpiles to Python

### HN body (paste into the first comment)

> Hi HN. Drift is a small language for building LLM agents. Instead of wiring up Python objects and async glue, you describe the agent in declarative blocks:
>
> ```
> agent GrantChecker {
>   model: prefer "claude-sonnet" fallback "gpt-4o"
>   budget: $0.10 per run
>
>   step assess(application: string) -> Decision {
>     let scored = rate application against rubric as confident<Decision>
>     if scored is confident { return scored.value }
>     fail "low confidence"
>   }
> }
> ```
>
> That's a full agent — model routing, budget cap, a typed intent verb, confidence-gated branching. The transpiler emits async Python that runs on a thin runtime handling provider failover, cost tracking, retries, schema validation, and memory.
>
> Built in (today): OpenAI + Anthropic providers, MCP tool support, parallel `for each`, persistent memory via Dendric, structured error handling, confidence gating, custom intent verbs, a formatter, a VS Code extension.
>
> What it's not: a general-purpose language. It's the orchestration glue around model calls — that's where the LOC budget reads worst in normal Python and the spec reads best.
>
> Honest about gaps: no LSP yet, type system is `confident<T>` plus structural types, source maps are coarse-grained. 350 tests, single-file LLM reference at /LLM.md for coding agents.
>
> Would love feedback on: the bet (orchestration as spec vs control flow), the surface area (which verb is missing for your use case), and the routing model.

### r/LocalLLaMA

Similar but slightly more skeptical audience. Lead with the cost/control hook:

> Built Drift — agents as declarative blocks, transpiles to async Python. Budget caps and cost tracking baked in; works with any OpenAI-compatible endpoint (including local via OPENAI_BASE_URL). 30-second `pip install drift-lang && drift new hello && drift run hello.drift`.

---

## 6. Optional: paid model warmup

Before posting publicly, run the canonical demo (`examples/inbox_triage_live.drift`) once against a real key and screenshot the cost report. The number ($0.0003 for a real agent) is the hook in any social post.

```bash
cd examples
/Users/rileycoleman/drift/.venv/bin/drift run inbox_triage_live.drift --input "$(cat sample_inbox.json)"
```

Save the screenshot. Embed in the social post.

---

## After everything ships

- [ ] PyPI: drift-lang searchable, `pip install` works
- [ ] VS Code marketplace: searchable, install works
- [ ] Domain points to landing page
- [ ] GitHub About has description, topics, website
- [ ] One social post live with cost-report screenshot
- [ ] README badge updated (add `[![PyPI](https://img.shields.io/pypi/v/drift-lang)](https://pypi.org/project/drift-lang/)`)

If anything in this checklist is unclear, ask Claude in your next session — the steps don't change, just paste them in.
