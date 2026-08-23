# ADR 0003: Randomness and Reproducibility

- Status: Accepted
- Date: 2026-08-23

## Context

Flip 7 simulation and later RL training depend on randomness. Hidden global
randomness makes rule debugging and experiment comparisons unreliable.

## Decision

Future simulation APIs accept an explicit seed or isolated RNG object. The game
engine owns simulation randomness; callers do not mutate global random state.
Tests use fixed seeds for deterministic transitions. Training and evaluation
record separate seeds, configurations, software versions, and checkpoints.

## Consequences

Runs can be reproduced and compared, while deliberately stochastic evaluation
can still use a documented seed set. Parallel environments must derive
independent child RNG streams rather than sharing mutable global state.
