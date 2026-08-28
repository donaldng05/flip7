# Configurations

Configuration files are version-controlled inputs for reproducible experiments.
They should contain small, reviewable YAML values and must not contain secrets.

Generated checkpoints, logs, metrics, and datasets belong in ignored artifact
directories or an explicitly configured external store. Use a descriptive run
name and record the configuration and seed with every future experiment.

`defaults.yaml` holds shared experiment inputs for the engine and environment
adapters. Training and evaluation schemas are added when those phases start.
`baselines.yaml` contains the reproducible Phase 4 agent and tournament
defaults.

`phase5.yaml` contains the initial PPO architecture, rollout, optimization,
opponent, and evaluation inputs. Generated checkpoints and results are written
to ignored artifact directories.

`phase6.yaml` contains the six-condition observation and learner-seat matrix.
Use `scripts/run_phase6.py` to train, evaluate, and summarize the matrix under
`artifacts/phase6/`.

`phase7.yaml` contains the baseline control, mixed policy league, latest-only
ablation, held-out evaluation, and Elo tournament inputs. Use
`scripts/run_phase7.py` to write the experiment under `artifacts/phase7/`.

`phase7-follow-up.yaml` contains the fixed-bank diversity screening, five-seed
confirmation, and evidence-gated MAPPO pilot. Use
`scripts/run_phase7_follow_up.py`; its outputs are isolated under
`artifacts/phase7-follow-up/`.
