"""Run a configured bounded Phase 5 PPO experiment."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from flip7.config.loader import load_config
from flip7.training import PPOConfig, PPOTrainer, write_history


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase5.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/ppo.pt"))
    parser.add_argument("--history", type=Path, default=Path("artifacts/training.json"))
    args = parser.parse_args()

    root = _mapping(load_config(args.config), "configuration")
    environment = _mapping(root["env"], "env")
    training = _mapping(root["training"], "training")
    config_values = dict(training)
    config_values.update(
        {
            "seed": int(root["seed"]),
            "player_count": int(root["players"]),
            "learner_id": int(environment["learner_id"]),
            "observation": str(environment["observation"]),
            "reward": str(environment["reward"]),
        }
    )
    config_values.pop("algorithm", None)
    config = PPOConfig(**config_values)
    opponents_config = _mapping(root["opponents"], "opponents")
    opponent_values = opponents_config["training"]
    if not isinstance(opponent_values, list):
        raise ValueError("opponents.training must be a list")
    opponents = tuple(str(name) for name in opponent_values)
    trainer = PPOTrainer(config, opponent_names=opponents)
    history = trainer.train(args.checkpoint)
    write_history(args.history, history)
    print(f"wrote checkpoint: {args.checkpoint}")
    print(f"wrote history: {args.history}")


if __name__ == "__main__":
    main()
