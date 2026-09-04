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
    aggregate_seed_summaries,
    bootstrap_rotated_uncertainty,
    classify_diagnostic,
    compare_seed_spreads,
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
DEFAULT_OUTPUT = Path("artifacts/phase7-resolve/calibration-v2.json")


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
    bootstrap_replicates: int,
) -> tuple[list[Any], dict[str, object]]:
    results = run_rotated_matchups(
        _cached_policy(checkpoint),
        baseline_factories(),
        games=games,
        seed_bases=seed_bases,
        observation=observation,
        matchups=PHASE6_MATCHUPS,
    )
    summary = summarize_rotated_results(results)
    summary["uncertainty"] = bootstrap_rotated_uncertainty(
        results,
        seed=sum((index + 1) * value for index, value in enumerate(seed_bases)),
        replicates=bootstrap_replicates,
    )
    return results, summary


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
    per_seed = _aggregate_runs_by_seed(runs)
    return {
        "runs": runs,
        "aggregate": summarize_rotated_results(results),
        "per_seed": per_seed,
    }


def _aggregate_runs_by_seed(
    runs: Sequence[Mapping[str, object]],
) -> dict[int, dict[str, object]]:
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for run in runs:
        seed = int(run["seed"])
        summary = _mapping(run["summary"], "run.summary")
        grouped.setdefault(seed, []).append(summary)
    return {
        seed: aggregate_seed_summaries(summaries)
        for seed, summaries in sorted(grouped.items())
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
    historical_seed_summaries = {
        int(item["seed"]): _mapping(item["summary"], "historical seed summary")
        for item in cast(list[object], reference_condition["seed_summaries"])
    }
    expected = _mapping(reference_condition["aggregate"], "basic_random aggregate")
    expected_win_share = float(expected["win_share"])
    expected_seat_spread = float(expected["seat_spread"])
    resolution = _mapping(root["resolution"], "resolution")
    tolerance = float(
        resolution.get(
            "reference_comparison_tolerance",
            resolution["control_reproduction_tolerance"],
        )
    )

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
                bootstrap_replicates=int(
                    resolution.get("uncertainty_replicates", 1000)
                ),
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
                    bootstrap_replicates=int(
                        resolution.get("uncertainty_replicates", 1000)
                    ),
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
    fresh_reference_by_seed = _aggregate_runs_by_seed(reference_runs)
    reference_reproduction = reference_reproduces(
        historical_seed_summaries,
        fresh_reference_by_seed,
        tolerance=tolerance,
    )
    gates = _mapping(root["gates"], "gates")
    max_seat_spread = float(gates["max_seat_spread"])
    reference_is_robust = all(
        float(summary["seat_spread"]) <= max_seat_spread
        for summary in fresh_reference_by_seed.values()
    )

    current_payload: dict[str, object] = {}
    for condition in args.conditions:
        batch_aggregates = [
            summarize_rotated_results(current_batch_results[condition][batch])
            for batch in range(args.repeats)
        ]
        payload = _condition_payload(
            current_runs[condition], current_all[condition]
        ) | {"batch_aggregates": batch_aggregates}
        candidate_by_seed = cast(dict[int, dict[str, object]], payload["per_seed"])
        payload["relative_to_reference"] = compare_seed_spreads(
            fresh_reference_by_seed,
            candidate_by_seed,
        )
        current_payload[condition] = payload

    control_comparisons = []
    if "balanced_control" in current_payload:
        control_payload = _mapping(
            current_payload["balanced_control"], "balanced_control payload"
        )
        control_comparisons = cast(
            list[Mapping[str, object]], control_payload["relative_to_reference"]
        )
    classification = classify_diagnostic(
        reference_valid=reference_reproduction,
        reference_is_robust=reference_is_robust,
        control_seed_comparisons=control_comparisons,
    )
    output: dict[str, object] = {
        "experiment": str(root["experiment"]),
        "diagnostic": "phase7-stability-calibration-v2",
        "protocol": {
            "games_per_seat": args.games_per_seat,
            "repeats": args.repeats,
            "training_seeds": training_seeds,
            "seed_bases": [list(batch) for batch in seed_batches],
            "seed_stride": args.seed_stride,
            "matchups": [list(pair) for pair in PHASE6_MATCHUPS],
            "phase6_reproduction_tolerance": tolerance,
            "calibration_rule": "per_seed_uncertainty_reference_envelope",
        },
        "provenance": provenance,
        "reference": {
            "expected": {
                "win_share": expected_win_share,
                "seat_spread": expected_seat_spread,
            },
            "historical_per_seed": historical_seed_summaries,
            "fresh_per_seed": fresh_reference_by_seed,
            "runs": reference_runs,
            "aggregate": reference_aggregate,
            "reproduces_phase6": reference_reproduction,
            "calibration_status": (
                "calibrated" if reference_reproduction else "recalibration-required"
            ),
            "target_met_per_seed": reference_is_robust,
            "batch_aggregates": reference_batch_aggregates,
        },
        "current": current_payload,
        "classification": classification,
        "interpretation": {
            "protocol_valid": reference_reproduction,
            "reference_not_robust": not reference_is_robust,
            "current_control_is_calibration_only": True,
            "phase7_status": "inconclusive",
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
