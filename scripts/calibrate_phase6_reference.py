"""Recalibrate the Phase 6 reference from an existing diagnostic audit."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

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
    reference_reproduces,
    validate_seed_plan,
)
from flip7.training import baseline_factories

DEFAULT_AUDIT = Path("artifacts/phase7-resolve/diagnostic-sparse-corrected.json")
DEFAULT_CONFIG = Path("configs/phase7-resolve.yaml")
DEFAULT_REFERENCE_SUMMARY = Path("artifacts/phase6/summary.json")
DEFAULT_OUTPUT = Path("artifacts/phase7-resolve/calibration-v2.json")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _read(path: Path, name: str) -> Mapping[str, object]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), name)


def _aggregate_runs_by_seed(
    runs: Sequence[Mapping[str, object]],
) -> dict[int, dict[str, object]]:
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for run in runs:
        grouped.setdefault(int(run["seed"]), []).append(
            _mapping(run["summary"], "run.summary")
        )
    return {
        seed: aggregate_seed_summaries(summaries)
        for seed, summaries in sorted(grouped.items())
    }


def _runs(value: object, name: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return [_mapping(item, name) for item in value]


def _reference_checkpoint(root: Path, seed: int) -> tuple[Path, ObservationFamily]:
    manifest_path = root / f"seed-{seed}" / "manifest.json"
    manifest = _read(manifest_path, "Phase 6 manifest")
    checkpoint_value = manifest.get("checkpoint")
    if not isinstance(checkpoint_value, str):
        raise ValueError(f"Phase 6 manifest is missing checkpoint for seed {seed}")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_file():
        checkpoint = manifest_path.parent / checkpoint
    if not checkpoint.is_file():
        raise ValueError(f"Phase 6 checkpoint does not exist for seed {seed}")
    observation_value = manifest.get("observation", "basic")
    if not isinstance(observation_value, str):
        raise ValueError(f"Phase 6 observation is invalid for seed {seed}")
    return checkpoint, ObservationFamily(observation_value)


def _reevaluate_reference(
    reference_root: Path,
    training_seeds: Sequence[int],
    seed_batches: Sequence[Sequence[int]],
    *,
    games: int,
    bootstrap_replicates: int,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    runs: list[Mapping[str, object]] = []
    provenance: list[Mapping[str, object]] = []
    for training_seed in training_seeds:
        checkpoint, observation = _reference_checkpoint(reference_root, training_seed)
        provenance.append(
            {
                "seed": training_seed,
                "checkpoint": str(checkpoint),
                "source": "immutable_phase6_checkpoint",
            }
        )

        def policy(path: Path = checkpoint) -> PPOAgent:
            return PPOAgent.from_checkpoint(path, deterministic=True)

        for batch, seed_bases in enumerate(seed_batches):
            results = run_rotated_matchups(
                policy,
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
            runs.append({"seed": training_seed, "batch": batch, "summary": summary})
    return runs, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument(
        "--reference-summary", type=Path, default=DEFAULT_REFERENCE_SUMMARY
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reference-root", type=Path, default=Path("artifacts/phase6/basic_random")
    )
    parser.add_argument(
        "--reevaluate-reference",
        action="store_true",
        help="freshly evaluate only immutable Phase 6 checkpoints",
    )
    parser.add_argument("--games-per-seat", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--seed-stride", type=int, default=None)
    parser.add_argument("--seed-bases", type=int, nargs=3, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument("--max-seat-spread", type=float, default=None)
    args = parser.parse_args()

    config = _mapping(load_config(args.config), "phase7-resolve configuration")
    resolution = _mapping(config["resolution"], "resolution")
    gates = _mapping(config["gates"], "gates")
    tolerance = float(
        args.tolerance
        if args.tolerance is not None
        else resolution.get("reference_comparison_tolerance", 0.01)
    )
    max_seat_spread = float(
        args.max_seat_spread
        if args.max_seat_spread is not None
        else gates.get("max_seat_spread", 0.05)
    )

    audit = _read(args.audit, "diagnostic audit")
    historical = _read(args.reference_summary, "Phase 6 summary")
    historical_conditions = _mapping(historical["conditions"], "Phase 6 conditions")
    historical_basic = _mapping(
        historical_conditions["basic_random"], "Phase 6 basic_random"
    )
    historical_seed_summaries = {
        int(item["seed"]): _mapping(item["summary"], "historical seed summary")
        for item in cast(list[object], historical_basic["seed_summaries"])
    }

    screening = _mapping(config["screening"], "screening")
    training_seeds = [int(seed) for seed in cast(list[object], screening["seeds"])]
    games = int(
        args.games_per_seat
        if args.games_per_seat is not None
        else resolution.get("diagnostic_games_per_seat", 1000)
    )
    repeats = int(
        args.repeats
        if args.repeats is not None
        else resolution.get("diagnostic_repeats", 2)
    )
    seed_stride = int(
        args.seed_stride
        if args.seed_stride is not None
        else resolution.get("diagnostic_seed_stride", 10000)
    )
    configured_bases = resolution.get("diagnostic_seed_bases", [21000, 22000, 23000])
    seed_bases = tuple(
        args.seed_bases if args.seed_bases is not None else int(value)
        for value in cast(list[object], configured_bases)
    )
    bootstrap_replicates = int(
        args.bootstrap_replicates
        if args.bootstrap_replicates is not None
        else resolution.get("uncertainty_replicates", 1000)
    )
    seed_batches = validate_seed_plan(
        training_seeds,
        seed_bases,
        games=games,
        repeats=repeats,
        seed_stride=seed_stride,
    )
    reference = _mapping(audit["reference"], "audit.reference")
    if args.reevaluate_reference:
        reference_runs, reference_provenance = _reevaluate_reference(
            args.reference_root,
            training_seeds,
            seed_batches,
            games=games,
            bootstrap_replicates=bootstrap_replicates,
        )
    else:
        reference_runs = _runs(reference["runs"], "audit.reference.runs")
        audit_provenance = _mapping(audit.get("provenance", {}), "audit.provenance")
        reference_provenance = [
            _mapping(item, "audit.reference provenance")
            for item in cast(list[object], audit_provenance.get("reference", []))
        ]
    fresh_reference = _aggregate_runs_by_seed(reference_runs)
    reference_reproduction = reference_reproduces(
        historical_seed_summaries,
        fresh_reference,
        tolerance=tolerance,
    )
    reference_is_robust = all(
        float(summary["seat_spread"]) <= max_seat_spread
        for summary in fresh_reference.values()
    )

    audit_current = _mapping(audit["current"], "audit.current")
    current: dict[str, object] = {}
    for condition, value in audit_current.items():
        payload = _mapping(value, f"audit.current.{condition}")
        runs = _runs(payload["runs"], f"audit.current.{condition}.runs")
        per_seed = _aggregate_runs_by_seed(runs)
        current[condition] = {
            "runs": runs,
            "aggregate": payload.get("aggregate"),
            "batch_aggregates": payload.get("batch_aggregates", []),
            "per_seed": per_seed,
            "relative_to_reference": compare_seed_spreads(fresh_reference, per_seed),
        }

    control_comparisons: Sequence[Mapping[str, object]] = ()
    if "balanced_control" in current:
        control = _mapping(current["balanced_control"], "balanced_control")
        control_comparisons = cast(
            Sequence[Mapping[str, object]], control["relative_to_reference"]
        )
    classification = classify_diagnostic(
        reference_valid=reference_reproduction,
        reference_is_robust=reference_is_robust,
        control_seed_comparisons=control_comparisons,
    )

    protocol = dict(_mapping(audit.get("protocol", {}), "audit.protocol"))
    protocol.update(
        {
            "calibration_rule": "per_seed_uncertainty_reference_envelope",
            "reference_comparison_tolerance": tolerance,
            "max_seat_spread_target": max_seat_spread,
            "reference_reevaluated": args.reevaluate_reference,
            "reference_games_per_seat": games,
            "reference_repeats": repeats,
            "reference_seed_batches": [list(batch) for batch in seed_batches],
            "uncertainty_replicates": bootstrap_replicates,
        }
    )
    output = {
        "experiment": "phase7-resolve",
        "diagnostic": "phase7-stability-calibration-v2",
        "protocol": protocol,
        "source": {
            "audit": str(args.audit),
            "reference_summary": str(args.reference_summary),
            "historical_artifacts_immutable": True,
            "reference_re_evaluation_only": args.reevaluate_reference,
        },
        "provenance": {
            "reference": reference_provenance,
            "audit": audit.get("provenance", {}),
        },
        "reference": {
            "historical_per_seed": historical_seed_summaries,
            "fresh_per_seed": fresh_reference,
            "runs": reference_runs,
            "reproduces_phase6": reference_reproduction,
            "calibration_status": (
                "calibrated" if reference_reproduction else "recalibration-required"
            ),
            "target_met_per_seed": reference_is_robust,
            "reference_is_not_robust": not reference_is_robust,
        },
        "current": current,
        "classification": classification,
        "interpretation": {
            "pooled_phase6_spread_is_retired": True,
            "phase7_status": "inconclusive",
            "potential_win_evaluation_allowed": False,
            "mappo_allowed": False,
            "phase8_unlocked": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote calibration: {args.output}")


if __name__ == "__main__":
    main()
