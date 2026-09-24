"""Run the reproducible Phase 6 state-information experiment matrix."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

from flip7 import __version__
from flip7.agents import PPOAgent
from flip7.config.loader import load_config
from flip7.envs import ObservationFamily
from flip7.evaluation import (
    run_rotated_matchups,
    summarize_rotated_results,
    write_phase6_results,
)
from flip7.experiment import (
    as_ints,
    as_list,
    as_mapping,
    as_pairs,
    as_strings,
    training_config,
    write_json,
)
from flip7.training import PPOTrainer, baseline_factories, write_history


def _run_condition(
    root: Mapping[str, object],
    condition: Mapping[str, object],
    seeds: Sequence[int],
    output_root: Path,
    *,
    workers: int = 1,
    rollout_workers: int = 1,
) -> dict[str, object]:
    condition_name = str(condition["name"])
    opponents = as_strings(
        as_mapping(root["opponents"], "opponents")["training"], "opponents.training"
    )
    evaluation = as_mapping(root["evaluation"], "evaluation")
    games = int(evaluation["games_per_seat"])
    seed_bases = as_ints(evaluation["seed_bases"], "evaluation.seed_bases")
    matchups = as_pairs(evaluation["matchups"], "evaluation.matchups")
    all_results = []
    seed_summaries: list[dict[str, object]] = []

    for seed in seeds:
        config = training_config(
            root,
            seed,
            observation_override=str(condition["observation"]),
            learner_seat_mode_override=str(condition["learner_seat_mode"]),
            training_override={"rollout_workers": rollout_workers},
        )
        run_dir = output_root / condition_name / f"seed-{seed}"
        checkpoint = run_dir / "checkpoint.pt"
        history_path = run_dir / "training.json"
        evaluation_path = run_dir / "evaluation.json"
        manifest_path = run_dir / "manifest.json"
        metadata: dict[str, object] = {
            "experiment": str(root["experiment"]),
            "condition": condition_name,
            "seed": seed,
            "observation": config.observation,
            "learner_seat_mode": config.learner_seat_mode,
            "package_version": __version__,
            "config": asdict(config),
            "opponents": list(opponents),
        }

        trainer = PPOTrainer(config, opponent_names=opponents)
        history = trainer.train(checkpoint)
        write_history(history_path, history)

        def policy(checkpoint_path: Path = checkpoint) -> PPOAgent:
            return PPOAgent.from_checkpoint(checkpoint_path, deterministic=True)

        results = run_rotated_matchups(
            policy,
            baseline_factories(),
            games=games,
            seed_bases=seed_bases,
            observation=ObservationFamily(config.observation),
            matchups=matchups,
            workers=workers,
            checkpoint_path=checkpoint,
        )
        write_phase6_results(
            evaluation_path,
            metadata
            | {
                "evaluation": {
                    "games_per_seat": games,
                    "seed_bases": list(seed_bases),
                    "matchups": [list(pair) for pair in matchups],
                }
            },
            results,
        )
        summary = summarize_rotated_results(results)
        write_json(
            manifest_path,
            metadata
            | {
                "checkpoint": str(checkpoint),
                "training_history": str(history_path),
                "evaluation": str(evaluation_path),
                "summary": summary,
            },
        )
        all_results.extend(results)
        seed_summaries.append({"seed": seed, "summary": summary})
        print(f"completed {condition_name} seed={seed}", flush=True)

    aggregate = summarize_rotated_results(all_results)
    return {
        "observation": str(condition["observation"]),
        "learner_seat_mode": str(condition["learner_seat_mode"]),
        "seeds": list(seeds),
        "seed_summaries": seed_summaries,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase6.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/phase6"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--rollout-workers", type=int, default=None)
    args = parser.parse_args()

    root = as_mapping(load_config(args.config), "configuration")
    eval_cfg = as_mapping(root.get("evaluation", {}), "evaluation")
    workers = (
        args.workers
        if args.workers is not None
        else int(cast(int | str, eval_cfg.get("workers", 1)))
    )
    if workers < 1:
        raise ValueError("workers must be positive")
    train_cfg = as_mapping(root.get("training", {}), "training")
    rollout_workers = (
        args.rollout_workers
        if args.rollout_workers is not None
        else int(cast(int | str, train_cfg.get("rollout_workers", 1)))
    )
    if rollout_workers < 1:
        raise ValueError("rollout_workers must be positive")
    seeds = as_ints(root["seeds"], "seeds")
    conditions = as_list(root["conditions"], "conditions")
    condition_results: dict[str, object] = {}
    for index, condition_value in enumerate(conditions):
        condition = as_mapping(condition_value, f"conditions[{index}]")
        if "name" not in condition:
            raise ValueError(f"conditions[{index}] is missing name")
        condition_results[str(condition["name"])] = _run_condition(
            root,
            condition,
            seeds,
            args.output_root,
            workers=workers,
            rollout_workers=rollout_workers,
        )

    reference = as_mapping(root["phase5_reference"], "phase5_reference")
    reference_win_share = float(reference["pooled_win_share"])
    reference_spread = float(reference["seat_spread"])
    minimum_improvement = float(reference["minimum_improvement"])
    maximum_spread_ratio = float(reference["maximum_spread_ratio"])
    for value in condition_results.values():
        condition = cast(dict[str, object], value)
        aggregate = cast(Mapping[str, object], condition["aggregate"])
        pooled_win_share = float(aggregate["win_share"])
        seat_spread = float(aggregate["seat_spread"])
        condition["improvement_vs_phase5"] = pooled_win_share - reference_win_share
        condition["seat_spread_ratio_vs_phase5"] = seat_spread / reference_spread
        condition["passes_gate"] = (
            condition["learner_seat_mode"] == "random"
            and pooled_win_share >= reference_win_share + minimum_improvement
            and seat_spread <= reference_spread * maximum_spread_ratio
        )

    random_conditions = {
        name: cast(Mapping[str, object], value)
        for name, value in condition_results.items()
        if cast(Mapping[str, object], value)["learner_seat_mode"] == "random"
    }
    best_name = max(
        random_conditions,
        key=lambda name: float(
            cast(Mapping[str, object], random_conditions[name])["aggregate"][
                "win_share"
            ]  # type: ignore[index]
        ),
    )
    summary = {
        "experiment": str(root["experiment"]),
        "phase5_reference": {
            "pooled_win_share": reference_win_share,
            "seat_spread": reference_spread,
            "minimum_improvement": minimum_improvement,
            "maximum_spread_ratio": maximum_spread_ratio,
        },
        "conditions": condition_results,
        "best_randomized_condition": best_name,
    }
    write_json(args.output_root / "summary.json", summary)
    print(f"wrote summary: {args.output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
