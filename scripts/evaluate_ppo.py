"""Evaluate a Phase 5 checkpoint against Phase 4 baseline pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

from flip7.agents import PPOAgent
from flip7.evaluation import run_matchup, write_results
from flip7.training import baseline_factories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--seed", type=int, default=700)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/evaluation.json")
    )
    args = parser.parse_args()

    def policy() -> PPOAgent:
        return PPOAgent.from_checkpoint(args.checkpoint, deterministic=True)

    roster = baseline_factories() | {"ppo": policy}
    results = [
        run_matchup(
            ("ppo", first, second),
            roster,
            games=args.games,
            seed=args.seed + index * 100,
        )
        for index, (first, second) in enumerate(
            (("random", "threshold"), ("risk", "ev"), ("dp", "threshold"))
        )
    ]
    write_results(args.output, results)
    print(f"wrote evaluation: {args.output}")


if __name__ == "__main__":
    main()
