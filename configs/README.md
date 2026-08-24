# Configurations

Configuration files are version-controlled inputs for reproducible experiments.
They should contain small, reviewable YAML values and must not contain secrets.

Generated checkpoints, logs, metrics, and datasets belong in ignored artifact
directories or an explicitly configured external store. Use a descriptive run
name and record the configuration and seed with every future experiment.

`defaults.yaml` holds shared experiment inputs for the engine and environment
adapters. Training and evaluation schemas are added when those phases start.
