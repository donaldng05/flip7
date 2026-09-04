# ADR 0009: Phase 7 follow-up diversity and MAPPO pilot

- Status: Accepted
- Date: 2026-08-27

## Context

Phase 7 league PPO cleared baseline robustness, but the mixed population was
only 0.56 points better than latest-only on held-out variants. One of three
seeds also regressed in the one-game tournament, so the adaptation conclusion
was too noisy and not stable across seeds.

## Decision

Keep the active feed-forward masked PPO architecture and add a deterministic
behavioural state bank, archive-all snapshot history, and explicit retention
ablations. Compare newest-only, temporal, and novelty-aware populations with
matched seeds and fixed final checkpoints. Use a paired ten-game tournament
with all seat permutations and direct update-10 comparisons.

Run a narrow native-AEC MAPPO pilot only after diversity-aware confirmation.
The pilot uses a shared basic-observation actor and centralized critic, with
the same optimizer budget and explicit seeds. MAPPO is adopted only if its
predeclared per-seed strength, spread, held-out, and stability criteria pass;
otherwise it remains deferred with the observed limitation recorded.

## Consequences

- Behavioural diversity is measured rather than inferred from snapshot count.
- Tournament ratings are less sensitive to the original one-game schedule.
- Existing Phase 6 and Phase 7 results remain reproducible reference data.
- The follow-up adds analysis and an evidence-gated MARL pilot without making
  centralized critics the default training path.
