# Phase 5 Initial RL Run

## Status

The Phase 5 implementation and bounded pipeline smoke run are complete. The
training pipeline works end to end: PPO updates, masked actions, checkpointing,
checkpoint restoration, and baseline evaluation all pass. The policy is not
yet a seat-robust competitive result.

## Implementation

- Algorithm: custom PyTorch PPO actor-critic
- Environment: `Flip7VsOpponentsEnv`, learner seat 0
- Observation: `deck_aware`
- Reward: `sparse_win`
- Player count: 3
- Training opponents: random, threshold, risk, expected-value, and DP
- Device: CPU
- Checkpoint: `artifacts/ppo.pt`
- Training history: `artifacts/training.json`
- Evaluation results: `artifacts/evaluation.json`

## Bounded run

The reproducible configuration is [`configs/phase5.yaml`](../configs/phase5.yaml)
with seed `7`, 10 updates, 256 rollout steps per update, four epochs, and
64-sample minibatches. The run completed successfully and produced finite
policy/value/entropy metrics. Re-loading the checkpoint restored both model
and optimizer state.

The three-game-per-matchup smoke evaluation produced the following win shares:

| Lineup | PPO | Other result |
| --- | ---: | --- |
| PPO / random / threshold | 33% | random 33%, threshold 33% |
| PPO / risk / expected-value | 0% | risk 100% |
| PPO / DP / threshold | 0% | threshold 100% |

The sample is intentionally too small for a performance claim. The policy
selected `hit` 464 times and `stay` 0 times across the three matchups,
indicating that substantially more training, reward/credit-assignment work,
or hyperparameter experimentation is needed before calling it competitive.
Those experiments are follow-up work, not evidence that the Phase 5 pipeline
is broken.

## Extended diagnostic run

Before Phase 6, three independent longer runs used seeds `7`, `17`, and `27`,
with 50 updates and 1,024 rollout steps per update (51,200 learner decisions
per seed). The policies learned to choose `stay` rather than selecting `hit`
exclusively.

With PPO fixed in player ID 0 and evaluated for 300 games per matchup, average
PPO win shares across the three seeds were:

| Matchup | PPO win share |
| --- | ---: |
| PPO / random / threshold | 66.2% |
| PPO / risk / expected-value | 46.4% |
| PPO / DP / threshold | 65.6% |

Those numbers are strongly position-dependent. In a paired seat-rotated
evaluation covering 2,700 games, PPO's pooled win shares were 36.8% against
random/threshold, 28.1% against risk/expected-value, and 36.3% against
DP/threshold. The risk and threshold baselines won 71.1% and 58.5–62.9% in
the corresponding rotated matchups. PPO's aggregate rotated seat spread was
approximately 40.4 percentage points.

The conclusion is qualitative rather than algorithmic: Phase 5 learning is
real, but the policy exploits favorable first-player conditions and remains
too bust-prone in later seats. Phase 6 therefore tests observation families
and randomized learner seating.
