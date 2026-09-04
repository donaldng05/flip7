"""Audit Phase 7 seat robustness using immutable artifacts and fresh seeds."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from flip7.agents import PPOAgent
from flip7.config.loader import load_config
from flip7.envs import ObservationFamily
from flip7.evaluation import (
    PHASE6_MATCHUPS,
    run_rotated_matchups,
    summarize_rotated_results,
)
from flip7.evaluation.diagnostics import (
    classify_diagnostic,
    config_fingerprint,
    reference_reproduces,
    resolve_manifest_files,
    validate_seed_plan,
)
from flip7.training import baseline_factories

CURRENT_REQUIRED_FILES = (
    "checkpoint",
    "training_history",
    "population",
    "diversity",
    "evaluation",
    "tournament",
    "manifest",
)
PHASE6_REQUIRED_FILES = ("checkpoint", "training_history", "evaluation")
DEFAULT_CURRENT_ROOT = Path("artifacts/phase7-follow-up-stability-recipe-scheduled")
DEFAULT_REFERENCE_ROOT = Path("artifacts/phase6/basic_random")
DEFAULT_REFERENCE_SUMMARY = Path("artifacts/phase6/summary.json")
DEFAULT_OUTPUT = Path("artifacts/phase7-resolve/diagnostic.json")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _read_mapping(path: Path, name: str) -> Mapping[str, object]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), name)


def _cached_policy(path: Path) -> Any:
    policy: PPOAgent | None = None

    def factory() -> PPOAgent:
        nonlocal policy
        if policy is None:
            policy = PPOAgent.from_checkpoint(path, deterministic=True)
        return policy

    return factory


def _manifest_path(root: Path, condition: str | None, seed: int) -> Path:
    candidates = []
    if condition is not None:
        candidates.extend(
            [
                root / condition / f"seed-{seed}" / "manifest.json",
                root / "screening" / condition / f"seed-{seed}" / "manifest.json",
            ]
        )
    else:
        candidates.append(root / f"seed-{seed}" / "manifest.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(
        "missing manifest for "
        f"condition={condition!r}, seed={seed}: {', '.join(map(str, candidates))}"
    )


def _manifest_record(
    root: Path,
    condition: str | None,
    seed: int,
    *,
    expected_update: int | None,
) -> tuple[Mapping[str, object], dict[str, Path]]:
    path = _manifest_path(root, condition, seed)
    manifest = _read_mapping(path, "manifest")
    required = (
        CURRENT_REQUIRED_FILES if condition is not None else PHASE6_REQUIRED_FILES
    )
    files = resolve_manifest_files(
        manifest,
        path,
        required_keys=required,
        expected_seed=seed,
        expected_update=expected_update,
    )
    return manifest, files


def _evaluate(
    checkpoint: Path,
    observation: ObservationFamily,
    *,
    games: int,
    seed_bases: Sequence[int],
) -> tuple[list[Any], dict[str, object]]:
    results = run_rotated_matchups(
        _cached_policy(checkpoint),
        baseline_factories(),
        games=games,
        seed_bases=seed_bases,
        observation=observation,
        matchups=PHASE6_MATCHUPS,
    )
    return results, summarize_rotated_results(results)


def _observation(manifest: Mapping[str, object], *, fallback: str) -> ObservationFamily:
    value = manifest.get("observation")
    if value is None:
        config = _mapping(manifest.get("config", {}), "manifest.config")
        value = config.get("observation", fallback)
    if not isinstance(value, str):
        raise ValueError("manifest observation must be a string")
    return ObservationFamily(value)


def _condition_payload(
    runs: list[dict[str, object]],
    results: Sequence[Any],
) -> dict[str, object]:
    return {
        "runs": runs,
        "aggregate": summarize_rotated_results(results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase7-resolve.yaml")
    )
    parser.add_argument("--current-root", type=Path, default=DEFAULT_CURRENT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument(
        "--reference-summary", type=Path, default=DEFAULT_REFERENCE_SUMMARY
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--games-per-seat", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument(
        "--seed-bases", type=int, nargs=3, default=(21000, 22000, 23000)
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=("balanced_control", "seat_aware_response_diverse"),
    )
    args = parser.parse_args()

    root = load_config(args.config)
    screening = _mapping(root["screening"], "screening")
    training_seeds = [int(seed) for seed in cast(list[object], screening["seeds"])]
    seed_batches = validate_seed_plan(
        training_seeds,
        args.seed_bases,
        games=args.games_per_seat,
        repeats=args.repeats,
        seed_stride=args.seed_stride,
    )
    reference_summary = _read_mapping(args.reference_summary, "Phase 6 summary")
    conditions = _mapping(reference_summary["conditions"], "Phase 6 conditions")
    reference_condition = _mapping(conditions["basic_random"], "basic_random")
    expected = _mapping(reference_condition["aggregate"], "basic_random aggregate")
    expected_win_share = float(expected["win_share"])
    expected_seat_spread = float(expected["seat_spread"])
    resolution = _mapping(root["resolution"], "resolution")
    tolerance = float(resolution["control_reproduction_tolerance"])

    reference_all: list[Any] = []
    reference_runs: list[dict[str, object]] = []
    current_all: dict[str, list[Any]] = {name: [] for name in args.conditions}
    current_runs: dict[str, list[dict[str, object]]] = {
        name: [] for name in args.conditions
    }
    current_batch_results: dict[str, dict[int, list[Any]]] = {
        name: {batch: [] for batch in range(args.repeats)} for name in args.conditions
    }
    provenance: dict[str, object] = {"reference": [], "current": {}}

    for seed in training_seeds:
        reference_manifest, reference_files = _manifest_record(
            args.reference_root, None, seed, expected_update=None
        )
        reference_provenance = {
            "seed": seed,
            "manifest": str(_manifest_path(args.reference_root, None, seed)),
            "checkpoint": str(reference_files["checkpoint"]),
            "config_fingerprint": config_fingerprint(reference_manifest),
        }
        cast(list[object], provenance["reference"]).append(reference_provenance)
        for batch, seed_bases in enumerate(seed_batches):
            results, summary = _evaluate(
                reference_files["checkpoint"],
                _observation(reference_manifest, fallback="basic"),
                games=args.games_per_seat,
                seed_bases=seed_bases,
            )
            reference_all.extend(results)
            reference_runs.append({"seed": seed, "batch": batch, "summary": summary})

        for condition in args.conditions:
            manifest, files = _manifest_record(
                args.current_root, condition, seed, expected_update=100
            )
            provenance_current = cast(dict[str, object], provenance["current"])
            provenance_current.setdefault(condition, [])
            cast(list[object], provenance_current[condition]).append(
                {
                    "seed": seed,
                    "manifest": str(_manifest_path(args.current_root, condition, seed)),
                    "checkpoint": str(files["checkpoint"]),
                    "config_fingerprint": config_fingerprint(manifest),
                }
            )
            for batch, seed_bases in enumerate(seed_batches):
                results, summary = _evaluate(
                    files["checkpoint"],
                    _observation(manifest, fallback="basic"),
                    games=args.games_per_seat,
                    seed_bases=seed_bases,
                )
                current_all[condition].extend(results)
                current_batch_results[condition][batch].extend(results)
                current_runs[condition].append(
                    {"seed": seed, "batch": batch, "summary": summary}
                )

    reference_aggregate = summarize_rotated_results(reference_all)
    reference_batch_aggregates: list[dict[str, object]] = []
    for batch in range(args.repeats):
        batch_runs = [run for run in reference_runs if run["batch"] == batch]
        summaries = [cast(Mapping[str, object], run["summary"]) for run in batch_runs]
        reference_batch_aggregates.append(_mean_summary(summaries))
    reference_reproduction = all(
        reference_reproduces(
            expected_win_share,
            expected_seat_spread,
            cast(Mapping[str, object], run["summary"]),
            tolerance=tolerance,
        )
        for run in reference_runs
    )

    current_payload: dict[str, object] = {}
    for condition in args.conditions:
        batch_aggregates = [
            summarize_rotated_results(current_batch_results[condition][batch])
            for batch in range(args.repeats)
        ]
        current_payload[condition] = _condition_payload(
            current_runs[condition], current_all[condition]
        ) | {"batch_aggregates": batch_aggregates}

    control_summary = (
        cast(Mapping[str, object], current_payload["balanced_control"])["aggregate"]
        if "balanced_control" in current_payload
        else None
    )
    control_batches = (
        cast(Mapping[str, object], current_payload["balanced_control"])[
            "batch_aggregates"
        ]
        if "balanced_control" in current_payload
        else []
    )
    gates = _mapping(root["gates"], "gates")
    classification = classify_diagnostic(
        reference_valid=reference_reproduction,
        control_summary=cast(Mapping[str, object] | None, control_summary),
        control_batch_summaries=cast(Sequence[Mapping[str, object]], control_batches),
        min_win_share=float(gates["min_baseline_win_share"]),
        max_seat_spread=float(gates["max_seat_spread"]),
    )
    output: dict[str, object] = {
        "experiment": str(root["experiment"]),
        "diagnostic": "phase7-stability-resolution",
        "protocol": {
            "games_per_seat": args.games_per_seat,
            "repeats": args.repeats,
            "training_seeds": training_seeds,
            "seed_bases": [list(batch) for batch in seed_batches],
            "seed_stride": args.seed_stride,
            "matchups": [list(pair) for pair in PHASE6_MATCHUPS],
            "phase6_reproduction_tolerance": tolerance,
        },
        "provenance": provenance,
        "reference": {
            "expected": {
                "win_share": expected_win_share,
                "seat_spread": expected_seat_spread,
            },
            "runs": reference_runs,
            "aggregate": reference_aggregate,
            "reproduces_phase6": reference_reproduction,
            "batch_aggregates": reference_batch_aggregates,
        },
        "current": current_payload,
        "classification": classification,
        "interpretation": {
            "protocol_valid": reference_reproduction,
            "current_control_is_calibration_only": True,
            "phase8_unlocked": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote diagnostic: {args.output}")


def _mean_summary(summaries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Average the scalar summary fields for a diagnostic batch."""
    if not summaries:
        raise ValueError("cannot average empty diagnostic summaries")
    win_share = sum(float(item["win_share"]) for item in summaries) / len(summaries)
    seat_spread = sum(float(item["seat_spread"]) for item in summaries) / len(summaries)
    return {
        "win_share": win_share,
        "seat_spread": seat_spread,
        "games": sum(int(item["games"]) for item in summaries),
        "per_seed_summaries": list(summaries),
    }


if __name__ == "__main__":
    main()
