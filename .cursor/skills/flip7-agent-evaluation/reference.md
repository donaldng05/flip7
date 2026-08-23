# Evaluation Record

Record:

```yaml
evaluation_name:
code_version:
seeds: []
games_per_matchup:
player_counts: []
agents: []
opponents:
  training_seen: []
  known_evaluation: []
  unseen: []
results_file:
```

## Report requirements

For every matchup, report game count, win rate, uncertainty interval or
standard error, average final score, round score, bust rate, Flip 7 frequency,
action frequencies, and player-count breakdown. State whether the result is
from training, known evaluation, or unseen opponents.

Do not call a policy generalizable when it was evaluated only against opponents
used during training.
