# Phase 5 Initial RL Run

## Status

The initial Phase 5 implementation and bounded pipeline smoke run are
complete. The first short policy is not yet a competitive result; it verifies
that PPO training, masked actions, checkpointing, and baseline evaluation work
end to end.

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
