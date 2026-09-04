# ADR 0010: Phase 7 Stability and Diversity-Benefit Evaluation

- Status: Accepted
- Date: 2026-08-28

## Context

The original Phase 7 follow-up showed a positive but sub-gate held-out margin
for novelty retention and substantial seed-to-seed variation in seat spread.
Random learner-seat sampling did not guarantee equal transition exposure, and a
broad population Elo was not a sufficiently direct measure of final-versus-
warmup adaptation.

## Decision

Add a separate stability branch and artifact namespace. Keep the Phase 6/7
baselines immutable and use balanced transition quotas for seats 0, 1, and 2.
Use a gradual learned-opponent curriculum with an explicit baseline floor and
warmup-anchor exposure. Compare latest-only, temporal, and response-diverse
retention under identical schedules, using only training-roster response
signatures for retention.

Use common-seed paired held-out evaluations and a focused multi-game tournament
for final-versus-warmup adaptation. Retain full-population Elo as a secondary
diagnostic. Evaluate one seat-aware fallback only when balanced-basic fails its
strength or seat-spread gate. Run MAPPO only after a confirmed seat-robust
league candidate is available.

## Consequences

- Seat spread is controlled during training rather than inferred from random
  episode counts.
- Held-out improvement can be attributed to retention and response coverage
  without using held-out opponents for selection.
- Tournament adaptation is measured directly with enough repeated games to
  reduce the original one-game noise.
- The seat-aware fallback and MAPPO remain evidence-gated and cannot silently
  replace the primary basic-observation path.
