"""Evaluate a Phase 5 checkpoint against Phase 4 baseline pairs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch

from flip7.agents import PPOAgent
from flip7.envs import ObservationFamily
from flip7.evaluation import (
    PHASE6_MATCHUPS,
    run_matchup,
    run_rotated_matchups,
    write_phase6_results,
    write_results,
)
from flip7.training import baseline_factories


def _checkpoint_observation(path: Path) -> ObservationFamily:
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    config = cast(Mapping[str, object], payload["config"])
    return ObservationFamily(str(config.get("observation", "deck_aware")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--seed", type=int, default=700)
    parser.add_argument(
        "--rotate-seats",
        action="store_true",
        help="evaluate PPO in all three seats with paired seed blocks",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/evaluation.json")
    )
    args = parser.parse_args()

    def policy() -> PPOAgent:
        return PPOAgent.from_checkpoint(args.checkpoint, deterministic=True)

    observation = _checkpoint_observation(args.checkpoint)
    baseline_roster = baseline_factories()
    if args.rotate_seats:
        results = run_rotated_matchups(
            policy,
            baseline_roster,
            games=args.games,
            seed_bases=tuple(
                args.seed + index * 1_000 for index in range(len(PHASE6_MATCHUPS))
            ),
            observation=observation,
        )
        write_phase6_results(
            args.output,
            {
                "checkpoint": str(args.checkpoint),
                "observation": observation.value,
                "evaluation_mode": "rotated_seats",
            },
            results,
        )
    else:
        roster = baseline_roster | {"ppo": policy}
        results = [
            run_matchup(
                ("ppo", first, second),
                roster,
                games=args.games,
                seed=args.seed + index * 100,
                observation=observation,
            )
            for index, (first, second) in enumerate(PHASE6_MATCHUPS)
        ]
        write_results(args.output, results)
    print(f"wrote evaluation: {args.output}")


if __name__ == "__main__":
    main()
