# ADR 0007: Custom PyTorch PPO for the initial learned agent

- Status: Accepted
- Date: 2026-08-23

## Context

Phase 5 needs a reproducible first learned policy while preserving the
environment and agent boundaries established in Phases 3 and 4. Flip 7 has a
variable legal-action mask, sparse game-level rewards, and a single learner
playing against frozen policies. The project does not yet need distributed
simulation, recurrent state, or an experiment-tracking service.

## Decision

Use a small custom PyTorch actor-critic implementation with PPO. The policy
will mask invalid discrete actions by setting their logits to negative
infinity before sampling or selecting an action. Training will use the
Gymnasium learner-seat wrapper and frozen Phase 4 baseline factories. The
native PettingZoo AEC environment remains the source of truth.

Checkpoints will contain model and optimizer state, PPO configuration,
observation/action dimensions, seed, opponent names, package version, and
training update. Training history and evaluation results will be JSON files in
ignored artifact directories.

## Consequences

- PPO behavior, masking, rollout boundaries, and checkpoint compatibility are
  explicit and directly testable.
- PyTorch becomes a runtime dependency when learning begins.
- The first implementation is intentionally CPU-friendly and single-process.
- Stable-Baselines3, Ray, vectorized environments, recurrent policies,
  self-play, and external tracking remain deferred until experiments justify
  them.
