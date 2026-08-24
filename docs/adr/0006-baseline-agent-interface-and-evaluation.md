# ADR 0006: Observation-only baseline agents and reproducible evaluation

## Status

Accepted.

## Context

Phase 4 needs interpretable opponents before any learned policy is introduced.
The environment already exposes stable observation vectors, discrete actions, and
action masks. Baselines must be comparable with future policies without
reimplementing the rules or reaching into the engine.

Risk-aware and algorithmic policies need remaining-card counts, while the
competitive observation intentionally does not expose them.

## Decision

Baseline policies implement the callable interface
`(observation, action_mask) -> discrete_action` and may consume only those
environment outputs. They must always return a currently masked action.

Randomness is supplied through isolated `random.Random` instances or explicit
seeds. Risk, expected-value, and dynamic-programming policies require the
`deck_aware` observation family. The DP baseline is bounded to the number-card
round subproblem; action-card targeting uses the shared deterministic target
policy rather than duplicating engine transitions.

Evaluation runs through `Flip7AECEnv` using fixed seed sets and records win
share, tie frequency, final score, round score, bust rate, Flip 7 frequency,
action frequencies, and player-count/matchup breakdowns. Generated result
files are experiment artifacts and remain outside source control by default.

## Consequences

- Baselines and future RL policies share one stable adapter-facing contract.
- Deck-informed strategies are explicit about their information advantage.
- The DP agent is useful and testable without claiming full-game optimality.
- Tournament results are reproducible and directly comparable across phases.
- Full-game simulation search and uncertainty intervals remain future evaluation
  extensions.
