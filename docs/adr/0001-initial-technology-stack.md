# ADR 0001: Initial Technology Stack

- Status: Accepted
- Date: 2026-08-23

## Context

The project is a Python research platform whose first priority is a correct,
reproducible simulator. It must support later RL and multi-agent work without
making the initial development environment unnecessarily heavy.

## Decision

Use Python 3.12+, `uv`, a `src/` layout, pytest, pytest-cov, Ruff, Pyright,
pre-commit, GitHub Actions, and optional Docker. Add NumPy, Gymnasium,
PettingZoo, PyTorch, Ray, Optuna, experiment tracking, and serving libraries
only when their roadmap phase has a concrete need.

## Consequences

The core remains quick to install and can run in CPU-only CI. Dependency choices
can be revisited when profiling or experiments provide evidence that a heavier
framework is valuable.
