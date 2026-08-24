# Contributing

## Development setup

Install Python 3.12 or newer and [uv](https://docs.astral.sh/uv/), then run:

```powershell
uv sync
uv run pre-commit install
```

Run the complete local quality gate with `scripts/check.ps1` on Windows or
`bash scripts/check.sh` on Linux/macOS.

Install the local commit hooks with `uv run pre-commit install`, or run them
manually across the repository with `uv run pre-commit run --all-files`.

## Engineering expectations

- Keep changes focused and explain the user or research problem they solve.
- Add or update tests for every behavior change.
- Write deterministic tests for game mechanics using explicit seeds.
- Keep the game engine, environment adapters, agents, training, and evaluation
  independently replaceable.
- Do not commit secrets, generated checkpoints, datasets, or experiment logs.
- Do not add a heavyweight dependency until a concrete milestone needs it.

## Changes and pull requests

Use a short-lived branch and a focused pull request. The pull request should
describe the motivation, summarize the change, list validation commands, and
call out any reproducibility or performance implications. All CI checks must
pass before merging.

Conventional Commit-style prefixes such as `feat:`, `fix:`, `test:`, `docs:`,
`refactor:`, and `chore:` are preferred.

## Current milestone

The current milestone is Phase 5 initial RL. Keep `flip7.core` free of RL
framework imports. Keep training configuration-driven and do not commit
generated checkpoints, logs, or experiment results.
