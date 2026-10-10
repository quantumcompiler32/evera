# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues in `quantumcompiler32/evera`. Use the `gh` CLI for all operations.

## Conventions

- **Create an issue**: `gh issue create --repo quantumcompiler32/evera --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --repo quantumcompiler32/evera --comments`, filtering comments by `jq` and also fetching labels.
- **List issues**: `gh issue list --repo quantumcompiler32/evera --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --repo quantumcompiler32/evera --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --repo quantumcompiler32/evera --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --repo quantumcompiler32/evera --comment "..."`

## Pull requests as a triage surface

**PRs as a request surface: no.**

When a skill says “publish to the issue tracker”, create a GitHub issue.

When a skill says “fetch the relevant ticket”, run `gh issue view <number> --repo quantumcompiler32/evera --comments`.
