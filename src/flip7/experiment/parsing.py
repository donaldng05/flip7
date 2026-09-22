"""Typed configuration parsing and collection validation for runners."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from flip7.training.mappo import MAPPOConfig
from flip7.training.ppo import PPOConfig


def as_mapping(value: object, name: str) -> Mapping[str, object]:
    """Ensure value is a dictionary mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def as_list(value: object, name: str) -> list[object]:
    """Ensure value is a list."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return cast(list[object], value)


def as_ints(value: object, name: str) -> tuple[int, ...]:
    """Ensure value is a list of integers, explicitly excluding booleans."""
    values = as_list(value, name)
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in values):
        raise ValueError(f"{name} must contain only integers")
    return tuple(cast(int, item) for item in values)


def as_strings(value: object, name: str) -> tuple[str, ...]:
    """Ensure value is a list of strings."""
    values = as_list(value, name)
    if not all(isinstance(item, str) for item in values):
        raise ValueError(f"{name} must contain only strings")
    return tuple(cast(str, item) for item in values)


def as_pairs(value: object, name: str) -> tuple[tuple[str, str], ...]:
    """Ensure value is a non-empty list of two-element string tuples."""
    pairs: list[tuple[str, str]] = []
    for index, item in enumerate(as_list(value, name)):
        pair = as_strings(item, f"{name}[{index}]")
        if len(pair) != 2:
            raise ValueError(f"{name} entries must contain two names")
        pairs.append((pair[0], pair[1]))
    if not pairs:
        raise ValueError(f"{name} must not be empty")
    return tuple(pairs)


def training_config(
    root: Mapping[str, object],
    seed: int,
    *,
    updates: int | None = None,
    observation_override: str | None = None,
    learner_seat_mode_override: str | None = None,
    training_override: Mapping[str, object] | None = None,
) -> PPOConfig:
    """Build a PPOConfig from root experiment configuration and overrides."""
    environment = as_mapping(root["env"], "env")
    training = dict(as_mapping(root["training"], "training"))
    training.pop("algorithm", None)
    if updates is not None:
        training["updates"] = updates
    if training_override is not None:
        training.update(training_override)

    def _opt_float(val: object | None) -> float | None:
        return float(cast(float | int | str, val)) if val is not None else None

    return PPOConfig(
        seed=seed,
        player_count=int(cast(int | str, root["players"])),
        learner_id=int(cast(int | str, environment["learner_id"])),
        learner_seat_mode=(
            learner_seat_mode_override
            or str(environment.get("learner_seat_mode", "fixed"))
        ),
        observation=observation_override or str(environment["observation"]),
        reward=str(training.get("reward", environment["reward"])),
        hidden_size=int(cast(int | str, training.get("hidden_size", 128))),
        rollout_steps=int(cast(int | str, training.get("rollout_steps", 256))),
        updates=int(cast(int | str, training.get("updates", 10))),
        epochs=int(cast(int | str, training.get("epochs", 4))),
        minibatch_size=int(cast(int | str, training.get("minibatch_size", 64))),
        learning_rate=float(
            cast(float | int | str, training.get("learning_rate", 3e-4))
        ),
        gamma=float(cast(float | int | str, training.get("gamma", 0.99))),
        gae_lambda=float(cast(float | int | str, training.get("gae_lambda", 0.95))),
        clip_epsilon=float(cast(float | int | str, training.get("clip_epsilon", 0.2))),
        value_coefficient=float(
            cast(float | int | str, training.get("value_coefficient", 0.5))
        ),
        entropy_coefficient=float(
            cast(float | int | str, training.get("entropy_coefficient", 0.01))
        ),
        max_grad_norm=float(
            cast(float | int | str, training.get("max_grad_norm", 0.5))
        ),
        device=str(training.get("device", "cpu")),
        network=str(training.get("network", "shared")),
        critic_seat_conditioned=bool(training.get("critic_seat_conditioned", False)),
        learning_rate_end=_opt_float(training.get("learning_rate_end")),
        entropy_coefficient_end=_opt_float(training.get("entropy_coefficient_end")),
        target_kl=_opt_float(training.get("target_kl")),
        value_clip_epsilon=_opt_float(training.get("value_clip_epsilon")),
    )


def mappo_config(
    root: Mapping[str, object],
    seed: int,
    *,
    updates: int | None = None,
) -> MAPPOConfig:
    """Build a MAPPOConfig from root experiment configuration."""
    environment = as_mapping(root["env"], "env")
    training = dict(as_mapping(root["training"], "training"))
    return MAPPOConfig(
        seed=seed,
        player_count=int(cast(int | str, root["players"])),
        observation=str(environment["observation"]),
        reward=str(training.get("reward", environment["reward"])),
        hidden_size=int(cast(int | str, training.get("hidden_size", 128))),
        rollout_steps=int(cast(int | str, training.get("rollout_steps", 1_024))),
        updates=int(
            cast(
                int | str,
                updates if updates is not None else training.get("updates", 100),
            )
        ),
        epochs=int(cast(int | str, training.get("epochs", 4))),
        minibatch_size=int(cast(int | str, training.get("minibatch_size", 64))),
        learning_rate=float(
            cast(float | int | str, training.get("learning_rate", 3e-4))
        ),
        gamma=float(cast(float | int | str, training.get("gamma", 0.99))),
        gae_lambda=float(cast(float | int | str, training.get("gae_lambda", 0.95))),
        clip_epsilon=float(cast(float | int | str, training.get("clip_epsilon", 0.2))),
        value_coefficient=float(
            cast(float | int | str, training.get("value_coefficient", 0.5))
        ),
        entropy_coefficient=float(
            cast(float | int | str, training.get("entropy_coefficient", 0.01))
        ),
        max_grad_norm=float(
            cast(float | int | str, training.get("max_grad_norm", 0.5))
        ),
        device=str(training.get("device", "cpu")),
    )
