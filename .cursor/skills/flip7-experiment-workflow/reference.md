# Experiment Run Record

Record these fields with every run:

```yaml
run_name:
code_version:
dependency_lock: uv.lock
config:
seeds: []
player_counts: []
observation:
reward:
training_opponents: []
evaluation_opponents: []
output_directory:
```

## Result summary

Report sample size and uncertainty with:

- game win rate
- average final and round scores
- bust rate
- Flip 7 frequency
- hit/stay and special-card action frequencies
- performance by player count
- performance by opponent class
- training efficiency and inference latency when relevant
