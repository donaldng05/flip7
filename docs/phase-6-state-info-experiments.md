# Phase 6 State Information and Seat-Robustness Experiments

## Objective

Determine whether richer public state information and training across all
learner seats improve the Phase 5 PPO policy's robustness. Phase 5 established
that the pipeline works and that the policy improves when starting first, but
not that it is seat-robust or competitive.

## Controlled experiment

All conditions use the same three-player environment, frozen Phase 4
opponents, custom PPO actor-critic, CPU device, `sparse_win` reward, and
training budget:

- Seeds: `7`, `17`, `27`
- 50 updates
- 1,024 rollout steps per update
- Four PPO epochs and 64-sample minibatches
- Opponents: random, threshold, risk, expected-value, and DP

The six-condition matrix is the factorial combination of:

| Observation | Fixed learner seat | Random learner seat |
| --- | ---: | ---: |
| `basic` | Yes | Yes |
| `competitive` | Yes | Yes |
| `deck_aware` | Yes | Yes |

The fixed `deck_aware` condition is the direct Phase 5 control. In random-seat
mode, a learner seat is sampled uniformly from player IDs 0–2 once per game
episode. The existing absolute-seat observation layout and ego one-hot remain
unchanged so seat randomization is isolated from a representation rewrite.

The primary observation ladder is:

1. `basic`: learner state and current round context.
2. `competitive`: public opponent state, scores, and turn/dealer context.
3. `deck_aware`: competitive state plus remaining card-face counts.

Historical observations, recurrent policies, ego-relative encodings, reward
shaping, self-play, and new opponent classes are deferred.

## Evaluation

Every checkpoint is evaluated against the same three Phase 5 matchup families:

- random / threshold
- risk / expected-value
- DP / threshold

PPO is placed in seats 0, 1, and 2. Each seat/matchup uses the same paired
100-game seed block across all conditions. Results include win share, average
final and round score, bust rate, Flip 7 frequency, tie frequency, action
counts, action shares, and approximate 95% confidence intervals.

Generated artifacts are written to:

```text
artifacts/phase6/<condition>/seed-<seed>/
    checkpoint.pt
    training.json
    evaluation.json
    manifest.json
artifacts/phase6/summary.json
```

Each manifest records the condition, seed, observation family, seat mode,
training configuration, opponent roster, package version, and artifact paths.

## Success gate

The paired rotated Phase 5 reference is approximately 33.7% pooled PPO win
share with a 40.4-point per-seat spread. A randomized-seat Phase 6 condition
is a robust improvement if it:

- reaches at least 38.7% pooled rotated win share; and
- reduces the per-seat spread to at most 30.3 points, a 25% relative reduction.

No condition is required to beat the risk heuristic in this phase. If no
condition passes, the ablation still identifies whether the next intervention
should be reward shaping, ego-relative encoding, or recurrent memory.

## Results

The full matrix completed on 2026-08-26: 18 training runs, each with 51,200
learner decisions, and 16,200 rotated evaluation games. The table reports
pooled PPO win share across all matchup families and seeds, per-seat win share,
and the Phase 6 gate result.

| Condition | Pooled PPO | Seat 0 | Seat 1 | Seat 2 | Spread | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `basic_fixed` | 64.5% | 70.2% | 66.2% | 57.1% | 13.1 pp | Control |
| `basic_random` | **65.4%** | 63.9% | 65.7% | 66.6% | **2.7 pp** | **Pass** |
| `competitive_fixed` | 28.7% | 62.3% | 9.6% | 14.1% | 52.7 pp | Fail |
| `competitive_random` | 35.7% | 35.6% | 36.9% | 34.4% | 2.5 pp | Fail strength |
| `deck_aware_fixed` | 33.8% | 60.6% | 20.4% | 20.2% | 40.4 pp | Control |
| `deck_aware_random` | 20.7% | 21.9% | 19.4% | 20.6% | 2.6 pp | Fail strength |

`basic_random` is the first condition to pass both Phase 6 criteria: it
improves pooled rotated win share by approximately 31.7 percentage points
over the 33.7% Phase 5 reference and reduces seat disparity by approximately
93%. It also averages about 200 final points with a 13.7% bust rate.

The result does not mean that less information is intrinsically better. Under
the fixed PPO architecture, sparse terminal reward, and identical 50-update
budget, the richer `competitive` and `deck_aware` inputs appear harder to
optimize. `competitive_random` removes the seat bias but remains just below
the 38.7% strength threshold; `deck_aware_random` is balanced but substantially
weaker. Baseline opponents continue to receive their original deck-aware views
in `basic` learner conditions, so the ablation changes only the learner's
available information.

The detailed per-seed artifacts and aggregate gate calculation are in
`artifacts/phase6/summary.json`.

## Reproduction

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\run_phase6.py
```

The configuration is recorded in [`configs/phase6.yaml`](../configs/phase6.yaml).
