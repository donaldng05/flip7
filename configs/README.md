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

`phase7-follow-up-stability.yaml` contains the seat-balanced PPO recipe,
response-diverse retention comparison, paired held-out evaluation, focused
adaptation tournament, and gated MAPPO stage. Use
`scripts/run_phase7_follow_up_stability.py`; outputs are isolated under
`artifacts/phase7-follow-up-stability/`.

`phase7-follow-up-stability-recipe.yaml` contains the opt-in stronger PPO
recipe: separate actor and critic features, a seat-conditioned value function,
scheduled learning rate and entropy, value clipping, KL diagnostics, and a
100-update screening budget. It uses the same stability runner and defaults to
the sparse-win evaluation protocol. Add `--training-reward potential_win` and
use a separate output root for the training-only potential-reward ablation.
For CPU screening, `--skip-full-population-tournament` omits only the
combinatorial secondary tournament; focused adaptation and all policy
evaluation remain enabled.

`phase7-resolve.yaml` is the isolated scientific-resolution protocol. It keeps
the stronger recipe and predeclared gates, while `scripts/diagnose_phase7_stability.py`
audits immutable prior artifacts and `scripts/compare_phase7_resolution.py`
performs paired sparse-versus-potential evaluation after corrected screening.
