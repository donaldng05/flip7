# ADR 0008: Population self-play and tournament evaluation

- Status: Accepted
- Date: 2026-08-26

## Context

Phase 6 established a seat-robust PPO policy against frozen baselines, but the
training interface supports only one active learner and one sampled heuristic
class per episode. Phase 7 needs competitive adaptation, diverse opponents,
historical policies, and a reproducible strength comparison without weakening
the Phase 6 control.

## Decision

Use league PPO as the first self-play method. One active feed-forward masked PPO
policy learns against independently sampled frozen baseline agents and learned
checkpoints. Learner seating is randomized per episode. Snapshots record their
training metadata and are bounded to a deterministic recent population.

Keep the existing callable agent contract and AEC source of truth. Add
per-seat opponent observation selection so basic learned policies and
deck-aware heuristics can share one game. Evaluate every three-player lineup
under all six seat permutations and calculate Elo through normalized pairwise
updates with explicit tie handling.

Defer MAPPO and other centralized-critic methods. Revisit them only after the
league has verified opponent coverage and either fails the Phase 7 robustness
gates or exhibits persistent non-transitive cycling or policy collapse that
historical snapshot sampling cannot address.

## Consequences

- Phase 6 PPO behavior, configuration, and artifacts remain reproducible.
- Population self-play is implementable with the existing PyTorch dependency.
- Historical snapshots and held-out parameter variants make opponent overfitting
  measurable rather than anecdotal.
- The approach does not yet solve simultaneous multi-agent credit assignment;
  that complexity remains an evidence-gated follow-up.
