"""Small, deterministic helpers for validating Phase 7 diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from flip7.evaluation.metrics import MatchupMetrics


def validate_seed_plan(
    training_seeds: Sequence[int],
    seed_bases: Sequence[int],
    *,
    games: int,
    repeats: int,
    seed_stride: int,
) -> tuple[tuple[int, ...], ...]:
    """Return fresh evaluation seed bases after rejecting overlapping blocks."""
    if games < 1 or repeats < 1:
        raise ValueError("games and repeats must be positive")
    if seed_stride < 1:
        raise ValueError("seed_stride must be positive")
    if not seed_bases or len(set(seed_bases)) != len(seed_bases):
        raise ValueError("evaluation seed bases must be unique")
    if any(base < 0 for base in seed_bases):
        raise ValueError("evaluation seed bases must be non-negative")

    intervals: list[tuple[int, int]] = []
    batches: list[tuple[int, ...]] = []
    for repeat in range(repeats):
        batch = tuple(base + repeat * seed_stride for base in seed_bases)
        batches.append(batch)
        intervals.extend((base, base + games - 1) for base in batch)

    intervals.sort()
    for (_, previous_end), (next_start, _next_end) in zip(
        intervals, intervals[1:], strict=False
    ):
        if next_start <= previous_end:
            raise ValueError("evaluation seed blocks overlap")
    training = set(training_seeds)
    if any(start <= seed <= end for start, end in intervals for seed in training):
        raise ValueError("evaluation seed blocks overlap training seeds")
    return tuple(batches)


def resolve_manifest_files(
    manifest: Mapping[str, object],
    manifest_path: Path,
    *,
    required_keys: Sequence[str],
    expected_seed: int | None = None,
    expected_update: int | None = None,
) -> dict[str, Path]:
    """Validate manifest file references and return resolved paths."""
    resolved: dict[str, Path] = {}
    for key in required_keys:
        value = manifest.get(key)
        if not isinstance(value, str):
            raise ValueError(f"manifest is missing a path for {key}")
        candidate = Path(value)
        if not candidate.is_file():
            candidate = manifest_path.parent / candidate
        if not candidate.is_file():
            raise ValueError(f"manifest path for {key} does not exist: {value}")
        resolved[key] = candidate

    if expected_seed is not None and manifest.get("seed") != expected_seed:
        raise ValueError(
            f"manifest seed {manifest.get('seed')!r} does not match {expected_seed}"
        )
    if (
        expected_update is not None
        and manifest.get("fixed_final_update") != expected_update
    ):
        raise ValueError(
            "manifest fixed_final_update does not match "
            f"{expected_update}: {manifest.get('fixed_final_update')!r}"
        )

    declared_hash = manifest.get("checkpoint_sha256")
    if declared_hash is not None:
        if not isinstance(declared_hash, str):
            raise ValueError("checkpoint_sha256 must be a string")
        digest = hashlib.sha256()
        with resolved["checkpoint"].open("rb") as checkpoint:
            for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != declared_hash:
            raise ValueError("checkpoint_sha256 does not match checkpoint contents")
    return resolved


def config_fingerprint(manifest: Mapping[str, object]) -> str | None:
    """Hash a manifest's recorded training configuration for provenance output."""
    config = manifest.get("config")
    if not isinstance(config, dict):
        return None
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def bootstrap_rotated_uncertainty(
    results: Sequence[MatchupMetrics],
    *,
    seed: int,
    replicates: int = 1_000,
) -> dict[str, object]:
    """Estimate rotated win-share uncertainty with a stratified bootstrap.

    The strata are the explicit matchup/learner-seat result blocks. Resampling
    within those blocks preserves the evaluation design while avoiding the
    invalid assumption that all seats share one independent pooled sample.
    """
    if not results:
        raise ValueError("cannot bootstrap an empty evaluation")
    if replicates < 100:
        raise ValueError("bootstrap replicates must be at least 100")

    strata: dict[int, list[list[float]]] = {0: [], 1: [], 2: []}
    for result in results:
        try:
            seat = result.agents.index("ppo")
        except ValueError as error:
            raise ValueError("evaluation result is missing the ppo agent") from error
        samples = result.win_share_samples.get("ppo", [])
        if not samples:
            raise ValueError("raw learner win-share samples are required for bootstrap")
        if seat not in strata:
            raise ValueError(f"unsupported learner seat: {seat}")
        strata[seat].append(list(samples))

    if any(not values for values in strata.values()):
        raise ValueError("evaluation is missing a learner seat")

    rng = random.Random(seed)
    seat_draws = [[0.0, 0.0, 0.0] for _ in range(replicates)]
    seat_intervals: dict[str, list[float]] = {}
    for seat, blocks in strata.items():
        total_games = sum(len(block) for block in blocks)
        point = 0.0
        for block in blocks:
            weight = len(block) / total_games
            point += math.fsum(block) / len(block) * weight
            for replicate in range(replicates):
                sample = rng.choices(block, k=len(block))
                seat_draws[replicate][seat] += math.fsum(sample) / len(block) * weight
        seat_intervals[str(seat)] = _percentile_interval(
            [draw[seat] for draw in seat_draws]
        )

    spread_draws = [max(draw) - min(draw) for draw in seat_draws]
    pooled_draws = [math.fsum(draw) / 3.0 for draw in seat_draws]
    return {
        "method": "stratified_game_bootstrap",
        "replicates": replicates,
        "seat_intervals": seat_intervals,
        "seat_spread_interval": _percentile_interval(spread_draws),
        "win_share_interval": _percentile_interval(pooled_draws),
    }


def _percentile_interval(values: Sequence[float]) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        return [0.0, 0.0]

    def quantile(probability: float) -> float:
        position = (len(ordered) - 1) * probability
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    return [quantile(0.025), quantile(0.975)]


def _number(value: object, fallback: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else fallback


def _integer(value: object, fallback: int = 0) -> int:
    return int(value) if isinstance(value, int) else fallback


def _interval(value: object, fallback: float) -> list[float]:
    if not isinstance(value, list):
        return [fallback, fallback]
    values = cast(list[object], value)
    if len(values) == 2:
        lower = _number(values[0], math.nan)
        upper = _number(values[1], math.nan)
        if math.isfinite(lower) and math.isfinite(upper) and lower <= upper:
            return [lower, upper]
    return [fallback, fallback]


def summary_interval(
    summary: Mapping[str, object], metric: str, *, seat: int | None = None
) -> list[float]:
    """Return a summary interval, including a legacy-artifact fallback."""
    if seat is not None:
        uncertainty = summary.get("uncertainty")
        if isinstance(uncertainty, dict):
            uncertainty_map = cast(Mapping[str, object], uncertainty)
            intervals = uncertainty_map.get("seat_intervals")
            if isinstance(intervals, dict):
                intervals_map = cast(Mapping[str, object], intervals)
                candidate = intervals_map.get(str(seat))
                if isinstance(candidate, list):
                    return _interval(cast(list[object], candidate), 0.0)
        per_seat = summary.get("per_seat")
        if isinstance(per_seat, dict):
            per_seat_map = cast(Mapping[str, object], per_seat)
            seat_summary = per_seat_map.get(str(seat))
            if isinstance(seat_summary, dict):
                seat_summary_map = cast(Mapping[str, object], seat_summary)
                value = _number(seat_summary_map.get("win_share"))
                return _interval(seat_summary_map.get("approximate_95_ci"), value)
        return [0.0, 0.0]

    value = _number(summary.get(metric))
    uncertainty = summary.get("uncertainty")
    if isinstance(uncertainty, dict):
        uncertainty_map = cast(Mapping[str, object], uncertainty)
        key = {
            "win_share": "win_share_interval",
            "seat_spread": "seat_spread_interval",
        }.get(metric)
        if key is not None:
            candidate = uncertainty_map.get(key)
            if isinstance(candidate, list):
                return _interval(cast(list[object], candidate), value)
    if metric == "win_share":
        return _interval(summary.get("approximate_95_ci"), value)
    if metric == "seat_spread":
        seat_intervals = [
            summary_interval(summary, "win_share", seat=seat) for seat in range(3)
        ]
        return [
            max(
                0.0,
                max(interval[0] for interval in seat_intervals)
                - min(interval[1] for interval in seat_intervals),
            ),
            max(interval[1] for interval in seat_intervals)
            - min(interval[0] for interval in seat_intervals),
        ]
    return [value, value]


def aggregate_seed_summaries(
    summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Pool repeated evaluation batches within one training seed.

    The returned seat spread is calculated after aggregating that seed's seat
    estimates. It is never calculated by pooling multiple training seeds.
    """
    if not summaries:
        raise ValueError("cannot aggregate empty seed summaries")
    weights = [max(1, _integer(summary.get("games"), 1)) for summary in summaries]
    total_weight = sum(weights)

    def weighted_value(metric: str, *, seat: int | None = None) -> float:
        values: list[float] = []
        for summary, weight in zip(summaries, weights, strict=True):
            if seat is None:
                value = _number(summary.get(metric))
            else:
                per_seat = cast(Mapping[str, object], summary.get("per_seat", {}))
                seat_summary = cast(Mapping[str, object], per_seat[str(seat)])
                value = _number(seat_summary.get("win_share"))
            values.append(value * weight)
        return sum(values) / total_weight

    def combined_interval(metric: str, *, seat: int | None = None) -> list[float]:
        point = weighted_value(metric, seat=seat)
        variance = 0.0
        for summary, weight in zip(summaries, weights, strict=True):
            interval = summary_interval(summary, metric, seat=seat)
            standard_error = (interval[1] - interval[0]) / (2.0 * 1.96)
            fraction = weight / total_weight
            variance += (fraction * standard_error) ** 2
        margin = 1.96 * math.sqrt(variance)
        lower_bound = point - margin
        upper_bound = point + margin
        if metric in {"win_share"} or seat is not None:
            lower_bound = max(0.0, lower_bound)
            upper_bound = min(1.0, upper_bound)
        return [lower_bound, upper_bound]

    per_seat: dict[str, dict[str, object]] = {}
    for seat in range(3):
        value = weighted_value("win_share", seat=seat)
        per_seat[str(seat)] = {
            "win_share": value,
            "approximate_95_ci": combined_interval("win_share", seat=seat),
        }
    seat_values = [_number(per_seat[str(seat)]["win_share"]) for seat in range(3)]
    spread = max(seat_values) - min(seat_values)
    seat_intervals = [
        cast(list[float], per_seat[str(seat)]["approximate_95_ci"]) for seat in range(3)
    ]
    spread_interval = [
        max(
            0.0,
            max(interval[0] for interval in seat_intervals)
            - min(interval[1] for interval in seat_intervals),
        ),
        max(interval[1] for interval in seat_intervals)
        - min(interval[0] for interval in seat_intervals),
    ]
    methods = [
        cast(Mapping[str, object], summary.get("uncertainty", {})).get("method")
        for summary in summaries
    ]
    method = (
        "stratified_game_bootstrap"
        if methods and all(item == "stratified_game_bootstrap" for item in methods)
        else "normal_summary_approximation"
    )
    return {
        "games": total_weight,
        "win_share": weighted_value("win_share"),
        "approximate_95_ci": combined_interval("win_share"),
        "per_seat": per_seat,
        "seat_spread": spread,
        "uncertainty": {
            "method": method,
            "seat_intervals": {
                seat: per_seat[seat]["approximate_95_ci"] for seat in per_seat
            },
            "seat_spread_interval": spread_interval,
            "win_share_interval": combined_interval("win_share"),
        },
    }


def _compatible(
    expected: Sequence[float], observed: Sequence[float], tolerance: float
) -> bool:
    return (
        observed[0] <= expected[1] + tolerance
        and observed[1] >= expected[0] - tolerance
    )


def reference_reproduces(
    expected_by_seed: Mapping[int, Mapping[str, object]],
    observed_by_seed: Mapping[int, Mapping[str, object]],
    *,
    tolerance: float,
) -> bool:
    """Check Phase 6 reproduction per training seed with uncertainty.

    Per-seed pooled win share is the reproduction check. The corresponding
    seat vectors and spreads are retained as the uncertainty-aware reference
    envelope rather than being treated as fixed historical point targets.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    expected_keys = set(expected_by_seed)
    observed_keys = set(observed_by_seed)
    if expected_keys != observed_keys or not expected_keys:
        return False
    for seed in sorted(expected_keys):
        expected = expected_by_seed[seed]
        observed = observed_by_seed[seed]
        if not _compatible(
            summary_interval(expected, "win_share"),
            summary_interval(observed, "win_share"),
            tolerance,
        ):
            return False
    return True


def compare_seed_spreads(
    reference_by_seed: Mapping[int, Mapping[str, object]],
    candidate_by_seed: Mapping[int, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Compare candidate/reference spread per seed using conservative intervals."""
    reference_keys = set(reference_by_seed)
    candidate_keys = set(candidate_by_seed)
    if reference_keys != candidate_keys:
        raise ValueError("reference and candidate seed sets must match")
    comparisons: list[dict[str, object]] = []
    for seed in sorted(reference_keys):
        reference = reference_by_seed[seed]
        candidate = candidate_by_seed[seed]
        reference_interval = summary_interval(reference, "seat_spread")
        candidate_interval = summary_interval(candidate, "seat_spread")
        delta_interval = [
            candidate_interval[0] - reference_interval[1],
            candidate_interval[1] - reference_interval[0],
        ]
        if delta_interval[1] < 0.0:
            relation = "better"
        elif delta_interval[0] > 0.0:
            relation = "worse"
        else:
            relation = "compatible"
        comparisons.append(
            {
                "seed": seed,
                "reference_spread": _number(reference.get("seat_spread")),
                "candidate_spread": _number(candidate.get("seat_spread")),
                "reference_interval": reference_interval,
                "candidate_interval": candidate_interval,
                "delta_interval": delta_interval,
                "relation": relation,
            }
        )
    return comparisons


def classify_diagnostic(
    *,
    reference_valid: bool,
    reference_is_robust: bool,
    control_seed_comparisons: Sequence[Mapping[str, object]],
) -> str:
    """Classify a diagnostic without treating a weak reference as a failure."""
    if not reference_valid:
        return "reference-recalibration-required"
    if not reference_is_robust:
        return "reference-not-robust / candidate-comparison-only"
    if not control_seed_comparisons:
        return "evaluation-noise/inconclusive"
    relations = [str(item.get("relation")) for item in control_seed_comparisons]
    if "worse" in relations:
        return "recipe-failure"
    if "compatible" in relations:
        return "evaluation-noise/inconclusive"
    return "protocol-valid"


__all__ = [
    "aggregate_seed_summaries",
    "bootstrap_rotated_uncertainty",
    "classify_diagnostic",
    "compare_seed_spreads",
    "config_fingerprint",
    "reference_reproduces",
    "resolve_manifest_files",
    "summary_interval",
    "validate_seed_plan",
]
