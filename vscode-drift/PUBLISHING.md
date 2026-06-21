# Publishing the Drift VS Code extension

The `.vsix` is built and lives next to this file. To get it on the Marketplace, you have two paths.

## Option A — `vsce publish` (one command, needs a PAT)

1. Create a Marketplace publisher named `rileyq7` at https://marketplace.visualstudio.com/manage
   (If you want a different name, update `publisher` in `package.json` and rebuild.)
2. Generate an Azure DevOps Personal Access Token with **Marketplace → Manage** scope:
   https://dev.azure.com → User Settings → Personal Access Tokens
3. Publish:
   ```bash
   cd vscode-drift
   npx @vscode/vsce login rileyq7   # paste PAT
   npx @vscode/vsce publish
   ```

## Option B — Upload the `.vsix` manually

1. Go to https://marketplace.visualstudio.com/manage/publishers/rileyq7
2. Click **New extension → Visual Studio Code**, upload `drift-lang-0.1.0.vsix`.

## Rebuilding after changes

```bash
cd vscode-drift
npx @vscode/vsce package --no-dependencies
```

This regenerates `drift-lang-<version>.vsix`. Bump `version` in `package.json` between publishes.
