"""Compare corrected sparse and potential-win checkpoints on paired games."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from flip7.config.loader import load_config
from flip7.envs import ObservationFamily
from flip7.evaluation import (
    PHASE6_MATCHUPS,
    run_paired_rotated_evaluation,
)
from flip7.evaluation.diagnostics import resolve_manifest_files
from flip7.experiment import as_mapping, cached_policy
from flip7.training import baseline_factories

REQUIRED_FILES = (
    "checkpoint",
    "training_history",
    "population",
    "diversity",
    "evaluation",
    "tournament",
    "manifest",
)


_mapping = as_mapping
_cached_policy = cached_policy


def _manifest_path(root: Path, condition: str, seed: int) -> Path:
    candidates = (
        root / condition / f"seed-{seed}" / "manifest.json",
        root / "screening" / condition / f"seed-{seed}" / "manifest.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"missing manifest for {condition} seed {seed}")


def _checkpoint(root: Path, condition: str, seed: int) -> tuple[Path, str]:
    path = _manifest_path(root, condition, seed)
    manifest = _mapping(
        json.loads(path.read_text(encoding="utf-8")), "resolution manifest"
    )
    files = resolve_manifest_files(
        manifest,
        path,
        required_keys=REQUIRED_FILES,
        expected_seed=seed,
        expected_update=100,
    )
    observation = manifest.get("observation")
    if not isinstance(observation, str):
        config = _mapping(manifest.get("config", {}), "manifest.config")
        observation = config.get("observation")
    if not isinstance(observation, str):
        raise ValueError("resolution manifest is missing observation")
    return files["checkpoint"], observation


def _aggregate_games(per_game: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not per_game:
        raise ValueError("paired evaluation produced no games")
    differences = [float(cast(float, game["difference"])) for game in per_game]
    by_seat: dict[str, list[float]] = {}
    by_matchup: dict[str, list[float]] = {}
    for game in per_game:
        seat = str(game["learner_seat"])
        matchup = str(game["matchup"])
        by_seat.setdefault(seat, []).append(float(cast(float, game["difference"])))
        by_matchup.setdefault(matchup, []).append(
            float(cast(float, game["difference"]))
        )

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values)

    return {
        "games": len(per_game),
        "mean_difference": mean(differences),
        "per_seat_mean_difference": {
            seat: mean(values) for seat, values in sorted(by_seat.items())
        },
        "per_matchup_mean_difference": {
            matchup: mean(values) for matchup, values in sorted(by_matchup.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase7-resolve.yaml")
    )
    parser.add_argument("--first-root", type=Path, required=True)
    parser.add_argument("--second-root", type=Path, required=True)
    parser.add_argument("--condition", default="balanced_response_diverse")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase7-resolve/paired-comparison.json"),
    )
    parser.add_argument("--games-per-seat", type=int, default=100)
    parser.add_argument(
        "--seed-bases", type=int, nargs=3, default=(21000, 22000, 23000)
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for evaluation (default: 1)",
    )
    args = parser.parse_args()
    if args.games_per_seat < 1:
        raise ValueError("games-per-seat must be positive")

    root = load_config(args.config)
    screening = _mapping(root["screening"], "screening")
    seeds = tuple(cast(int, seed) for seed in cast(list[object], screening["seeds"]))
    all_games: list[Mapping[str, object]] = []
    per_seed: list[dict[str, object]] = []
    first_observation: str | None = None
    second_observation: str | None = None
    for seed in seeds:
        first_checkpoint, first_observation = _checkpoint(
            args.first_root, args.condition, seed
        )
        second_checkpoint, second_observation = _checkpoint(
            args.second_root, args.condition, seed
        )
        if first_observation != second_observation:
            raise ValueError("paired checkpoints must use the same observation family")
        seed_bases = tuple(base + seed * 1000 for base in args.seed_bases)
        paired = run_paired_rotated_evaluation(
            _cached_policy(first_checkpoint),
            _cached_policy(second_checkpoint),
            baseline_factories(),
            games=args.games_per_seat,
            seed_bases=seed_bases,
            observation=ObservationFamily(cast(str, first_observation)),
            matchups=PHASE6_MATCHUPS,
            first_name="corrected_sparse",
            second_name="potential_win",
            workers=args.workers,
            checkpoint_paths=(first_checkpoint, second_checkpoint),
        )
        payload = paired.as_dict()
        per_game = cast(list[Mapping[str, object]], payload["per_game"])
        all_games.extend(per_game)
        per_seed.append(
            {"seed": seed, **payload, "aggregate": _aggregate_games(per_game)}
        )

    output = {
        "experiment": str(root["experiment"]),
        "condition": args.condition,
        "first_root": str(args.first_root),
        "second_root": str(args.second_root),
        "first": "corrected_sparse",
        "second": "potential_win",
        "games_per_seat": args.games_per_seat,
        "matchups": [list(pair) for pair in PHASE6_MATCHUPS],
        "seeds": list(seeds),
        "per_seed": per_seed,
        "aggregate": _aggregate_games(all_games),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote paired comparison: {args.output}")


if __name__ == "__main__":
    main()
