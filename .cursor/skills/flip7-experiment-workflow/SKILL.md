---
name: flip7-experiment-workflow
description: Design and run reproducible Flip 7 training or research experiments from versioned configurations. Use when starting, changing, or documenting an experiment.
disable-model-invocation: true
---

# Flip 7 Experiment Workflow

Use this skill for training, ablations, self-play, or controlled simulation
experiments.

1. Start from a version-controlled config under `configs/`; do not bury
   hyperparameters in source code.
2. Record commit/version, `uv.lock`, config path, seeds, player count,
   observations, rewards, opponent classes, and output location.
3. Keep training opponents separate from known and unseen evaluation opponents.
4. Run a small deterministic smoke run with bounded resources before scaling.
5. Save machine-readable results and metadata; keep generated artifacts outside
   Git unless intentionally curated.
6. Report win rate, average final score, round score, bust rate, Flip 7
   frequency, action distribution, and performance by player count.

Do not assume a particular RL framework or tracking service until the project
phase selects one. Use [reference.md](reference.md) for the run record template.
