# Phase 7 Stability and Diversity-Benefit Follow-Up

## Purpose

This experiment addresses the two unresolved Phase 7 findings: the
behaviourally novel league was only 2.17 points better than latest-only on
held-out variants, and tournament adaptation was not seat-stable across seeds.
The existing Phase 6, Phase 7, and Phase 7 follow-up results remain immutable
reference data. New outputs are written under
`artifacts/phase7-follow-up-stability/`.

## Training design

The primary policy remains the feed-forward masked PPO policy with the Phase 6
`basic` observation. Each PPO update collects equal transition quotas for
learner seats 0, 1, and 2. The stability recipe uses 1,026 rollout steps,
which divides evenly across the three seats, a learning rate of `0.00015`, and
entropy coefficient `0.015`.

League exposure is baseline-only through update 10. Learned-opponent exposure
then ramps from 25% to 60% over 25 updates while maintaining a 40% baseline
floor. The update-10 policy receives explicit anchor exposure, snapshots are
archived every two updates, and selection remains without replacement when
possible.

The response-diverse condition uses training-roster response signatures in
addition to fixed-bank masked action distributions. Held-out variants are
never used for retention or training decisions. Latest-only, temporal, and
response-diverse conditions share the same training recipe and exposure
schedule; only retention differs.

If the balanced `basic` response-diverse policy fails baseline strength or
seat-spread gates, the runner automatically evaluates one `seat_aware`
fallback. The fallback is not evaluated or promoted when balanced-basic passes.

## Evaluation protocol

Held-out comparisons use common game-seed blocks and report paired game-level
and block-level confidence intervals. Confirmation uses seeds 7, 17, 27, 37,
and 47, 100 updates, and 200 games per learner seat. The final update-100
checkpoint is the only primary candidate.

Adaptation is measured with a focused final-versus-update-10 tournament using
all six seat permutations and 20 games per lineup. The full population Elo is
retained as a secondary diagnostic. Both report simultaneous normalized
pairwise Elo updates with initial rating 1500 and `K=32`.

Run screening with:

```powershell
uv run python scripts/run_phase7_follow_up_stability.py --stage screening --workers 8
```

Run confirmation after screening with:

```powershell
uv run python scripts/run_phase7_follow_up_stability.py --stage confirmation --workers 8
```

Use `--resume` after an interrupted run. The targeted MAPPO pilot may be run
only after a confirmation summary reports a seat-robust stable adaptation
pass:

```powershell
uv run python scripts/run_phase7_follow_up_stability.py --stage mappo
```

## Gates

The diversity claim requires at least a three-point held-out gain over matched
latest-only, a paired interval excluding zero, no confirmation seed more than
two points below latest-only, and materially greater active-population response
diversity than latest-only and temporal retention.

Stable adaptation requires every confirmation seed to reach at least 60.4%
baseline win share, no more than five points of seat spread, final Elo of at
least 1500, at least 25 Elo over update 10, and positive paired final-versus-
warmup improvement. Failed gates remain diagnostic; no intermediate checkpoint
is substituted after evaluation.

## Artifacts

Each run contains `checkpoint.pt`, `training.json`, `population.json`,
`diversity.json`, `evaluation.json`, `tournament.json`, and `manifest.json`.
The population artifact records seat quotas, reset seeds, exposure, anchors,
retention, and response signatures. The diversity artifact records archived
and active behavior metrics, matchup matrices, cycles, and focused/full Elo.
