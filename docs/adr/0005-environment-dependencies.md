# ADR 0005: NumPy, Gymnasium, and PettingZoo are environment dependencies

## Status

Accepted.

## Context

Engineering standards keep `flip7.core` on the standard library and delay RL
libraries until the environment adapter exists. Observations and spaces need
NumPy arrays. PettingZoo is built on Gymnasium spaces.

## Decision

Add `numpy`, `gymnasium`, and `pettingzoo` as main package dependencies, locked
through `uv.lock`. Only `flip7.envs` may import them. `flip7.core` remains free
of those imports and continues to own rules, state, and seeded RNG objects.

Do not add SuperSuit, PyTorch, Stable-Baselines3, Ray, Optuna, or experiment
trackers in Phase 3.

## Consequences

- CI installs the environment stack on every run.
- A core-only import path stays available for rules tests.
- Training-library choice remains a Phase 5 ADR.
