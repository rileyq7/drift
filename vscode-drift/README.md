# Drift for VS Code

Syntax highlighting for `.drift` files — the [Drift](https://github.com/rileyq7/drift) intent-based language for agentic systems.

Highlights:

- Declaration keywords (`agent`, `step`, `state`, `model`, `tool`, `memory`, `pipeline`)
- Intent verbs (`summarize`, `classify`, `extract`, `generate`, …)
- Routing keywords (`confident`, `prefer`, `fallback`, `upgrade when`, `stream … then …`)
- Memory keywords (`remember`, `recall`, `deja_vu`, `forget`, `tagged`, `older than`)
- Error handling (`attempt`, `recover`, `retry`, `fail`)
- Currency (`$0.10`) and duration (`30s`, `5m`, `1h`) literals
- Pipeline operators (`->`, `=>`, `~>`, `|>`)
- String interpolation (`"Hello, {name}!"`)

## Local development

```bash
code --extensionDevelopmentPath=./vscode-drift
```

Open any `.drift` file in the new window — highlighting should activate automatically.
