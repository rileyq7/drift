# Drift Cookbook

Copy-paste recipes. Each is a complete, runnable Drift program.

## 1. Hello agent

The smallest useful program.

```drift
config {
  name: "Hello"
  version: "0.1.0"
}

agent Greeter {
  model: "gpt-5.4-nano"
  budget: $0.01 per run

  step greet(name: string) -> string {
    let message = generate a warm one-sentence greeting using name as string
    respond "Hello, {name}!"
    return message
  }
}
```

Run: `drift run hello.drift --input '{"name":"Riley"}'`

## 2. Confidence-gated escalation

Cheap model for the easy cases, escalate when unsure.

```drift
schema Decision {
  approved: bool
  reasoning: string
  confidence: number between 0 and 1
}

agent GrantChecker {
  model {
    default: "claude-haiku"
    upgrade to "claude-sonnet" when {
      confidence < 0.7
    }
  }
  budget: $0.50 per run

  step assess(application: string) -> Decision {
    let scored = rate application against grant_criteria as confident<Decision>
    if scored is confident {
      return scored.value
    }
    escalate to human review
  }
}
```

## 3. Parallel triage

Classify a batch of items concurrently.

```drift
schema EmailAnalysis {
  subject: string
  priority: one of "urgent", "normal", "low"
  category: one of "work", "personal", "newsletter", "spam"
  summary: string
}

agent InboxSorter {
  model: "gpt-4o-mini"
  budget: $0.20 per run

  step sort(emails: list<string>) -> list<EmailAnalysis> {
    let results = []
    for each email in emails parallel {
      let analysis = classify email as EmailAnalysis
      if analysis.priority == "urgent" {
        respond "URGENT: {analysis.subject}"
      }
      results.add(analysis)
    }
    return results
  }
}
```

## 4. Retry with structured recovery

Handle rate limits and budget caps cleanly.

```drift
agent ResilientFetcher {
  model: "claude-haiku"
  budget: $0.10 per run

  step fetch_and_summarize(url: string) -> string {
    attempt {
      let content = http.get(url)
      return summarize content as string
    }
    recover on rate limited { retry }
    recover on budget exceeded {
      fail "ran out of budget — try a smaller input"
    }
    recover on failure in fetch_and_summarize {
      respond "couldn't reach {url}, returning stub"
      return "summary unavailable"
    }
  }
}
```

## 5. Memory-aware advisor

Carries context across runs.

```drift
agent Advisor {
  model: "claude-haiku"
  memory: dendric("user_123")
  budget: $0.20 per run

  step advise(question: string) -> string {
    let context = recall question for "advice"
    let answer = answer question using context as string
    remember answer tagged "advice", "user_123"
    return answer
  }
}
```

## 6. MCP tool use

Read a file via an MCP server, summarize it.

```drift
tool fs from mcp "stdio:/usr/local/bin/mcp-fs"

agent DocSummarizer {
  model: "claude-haiku"
  budget: $0.05 per run

  step summarize_file(path: string) -> string {
    let content = fs.read_file(path: path)
    return summarize content as string
  }
}
```

## 7. REST tool inline

Declare a GitHub client without a Python module.

```drift
tool github {
  endpoint: "https://api.github.com"
  auth: env("GITHUB_TOKEN")
  action list_issues(repo: string) -> list<dict> {
    GET "/repos/{repo}/issues"
  }
}

agent IssueTriage {
  model: "claude-haiku"
  budget: $0.10 per run

  step triage(repo: string) -> list<string> {
    let issues = github.list_issues(repo: repo)
    let urgent = []
    for each issue in issues parallel {
      let analysis = classify issue.title as Priority
      if analysis.level == "urgent" {
        urgent.add(issue.number)
      }
    }
    return urgent
  }
}

schema Priority {
  level: one of "urgent", "normal", "low"
}
```

## 8. Multi-agent pipeline

A clean fan-out with one source-of-truth flow definition.

```drift
agent Tagger { ... }
agent Router { ... }
agent Notifier { ... }

pipeline triage {
  input -> Tagger.tag -> Router.route -> Notifier.send
  Tagger.tag ~> Logger.log
}
```

## 9. Custom intent verb

When the built-ins don't fit your domain.

```drift
define verb evaluate {
  default output: ScoreReport
  clauses: against, considering, with
  prompt: "Evaluate the input against the supplied criteria. Score and explain."
}

schema ScoreReport {
  score: number between 0 and 100
  reasoning: string
  confidence: number between 0 and 1
}

agent Reviewer {
  model: "claude-sonnet"
  budget: $0.50 per run

  step review(submission: string) -> ScoreReport {
    return evaluate submission against rubric considering style, clarity as ScoreReport
  }
}
```

## 10. Stream-then for snappy UX

Fast preview, slow reasoning. The fast model's tokens stream first, then the slow model's full reasoning replaces them.

```drift
agent ChatBot {
  model: stream "claude-haiku" then "claude-sonnet"
  budget: $0.10 per run

  step respond_to(question: string) -> string {
    return answer question using "documentation context" as string
  }
}
```
