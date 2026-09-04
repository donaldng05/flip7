# Phase 7 Follow-Up: Population Diversity, Stable Adaptation, and MAPPO

## Purpose

The original Phase 7 implementation was complete but its scientific gates
were not. `league_mixed` was robust against the Phase 6 baseline roster, yet
its held-out advantage over `latest_only` was only 0.56 points. Tournament
adaptation was also seed-sensitive: seed 27 declined by 99.4 Elo. This
follow-up tests whether behavioural population diversity and a paired
tournament protocol resolve those limitations before Phase 8.

The original Phase 7 configuration, runner, and artifacts remain immutable.
Follow-up outputs are isolated under `artifacts/phase7-follow-up/`.

## Experimental controls

All follow-up PPO conditions use the Phase 6 `basic` observation, randomized
learner seating, explicit CPU seeds, `sparse_win`, 1,024 rollout steps, four
epochs, and 64-sample minibatches. Baseline opponents retain their existing
deck-aware observations. The fixed state bank contains 2,048 legal decision
states generated from seed 70,000 and is reused for every policy and seed.

The screening conditions are:

| Condition | Retention | Learned exposure |
| --- | --- | ---: |
| `latest_only_dense` | newest snapshot only | 75% |
| `temporal_dense` | 12 temporally distributed snapshots | 75% |
| `novelty_dense` | farthest-point behavioural selection | 75% |

Snapshots are archived at update 10 and every two updates thereafter. The
update-10 warmup anchor is retained independently. Novelty selection reserves
early, midpoint, and final anchors, then greedily maximizes minimum pairwise
Jensen-Shannon divergence, using update recency only as a deterministic tie
break. Opponent slots sample independently and avoid replacement when enough
entries exist.

Run screening with:

```powershell
uv run python scripts/run_phase7_follow_up.py --stage screening --tournament-workers 8
```

Tournament games are independent, so the runner can execute them in parallel
on CPU while collecting results in the fixed schedule order. The worker count
does not change game seeds, metrics, or Elo updates. If a long run is
interrupted, add `--resume` to reuse only complete manifest-validated runs and
continue the missing conditions.

The confirmation matrix uses seeds 7, 17, 27, 37, and 47, 100 updates, 200
games per rotated seat, and ten games per tournament lineup. The final update
100 checkpoint is always evaluated; intermediate checkpoints are diagnostic
only. Use `--stage confirmation` after the predeclared screening choice has
been recorded in the experiment notes.

## Measurements and gates

`diversity.json` evaluates every archived checkpoint, including the warmup
anchor, on the fixed bank. It records masked action distributions, mean JS
divergence, deterministic action disagreement, entropy, action support,
conditional matchup win-share matrices, Elo progression, and detected
three-policy cycles. `population.json` records archive/active membership,
anchor identity, exposure counts, and retention configuration.

The population-diversity claim requires a minimum three-point held-out win
share advantage over matched latest-only, a paired 95% interval excluding zero,
and no confirmation seed more than two points below latest-only. The retained
population must also be behaviourally more diverse than latest-only and
temporal retention.

Stable adaptation requires, for every confirmation seed, at least 60.4% pooled
baseline win share, at most five points of seat spread, final Elo at least
1500, and at least 25 Elo over the fixed update-10 anchor. No favourable
intermediate checkpoint may replace the final update.

The paired tournament uses a participant-order-independent seed schedule, all
six seat permutations, ten games per lineup, simultaneous normalized pairwise
Elo updates (`1500`, `K=32`), and 0.5 for ties. Direct final-versus-warmup
comparisons are reported alongside the full population tournament.

## MAPPO pilot

MAPPO runs only after the diversity-aware confirmation is available. It uses
the native AEC environment, three trainable seats, a shared masked feed-forward
actor over the basic observation, and a centralized critic over the three
per-seat observations, active-seat one-hot vector, and normalized round
metadata. Each seat receives its own sparse-win reward. The pilot matches the
PPO budget and uses seeds 7, 17, and 27.

MAPPO is adopted only if every pilot seed clears the strength and seat-spread
gates, held-out performance improves by at least three points over matched
latest-only or worst-seed tournament instability is reduced by at least 50%
without losing more than one pooled strength point, and the result is
reproducible. Otherwise the ADR records an evidence-based deferral.

## Artifact contract

Each trained condition writes `checkpoint.pt`, `checkpoints/`,
`training.json`, `population.json`, `diversity.json`, `evaluation.json`,
`tournament.json`, and `manifest.json` beneath the follow-up artifact root.
The manifest records configuration, fixed final update, checkpoint hash, state
bank metadata, schedule size, and every artifact path.

## Results

The quantitative results are intentionally populated only after the fixed
screening, confirmation, and MAPPO commands complete. Failed gates are
reported as diagnostic findings; the runner does not select a favourable
checkpoint after seeing evaluation results.
