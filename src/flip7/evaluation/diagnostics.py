"""Small, deterministic helpers for validating Phase 7 diagnostics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast


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


def reference_reproduces(
    expected_win_share: float,
    expected_seat_spread: float,
    observed: Mapping[str, object],
    *,
    tolerance: float,
) -> bool:
    """Check the pre-registered Phase 6 reproduction tolerance."""
    return (
        abs(float(cast(float, observed["win_share"])) - expected_win_share) <= tolerance
        and abs(float(cast(float, observed["seat_spread"])) - expected_seat_spread)
        <= tolerance
    )


def classify_diagnostic(
    *,
    reference_valid: bool,
    control_summary: Mapping[str, object] | None,
    control_batch_summaries: Sequence[Mapping[str, object]],
    min_win_share: float = 0.604,
    max_seat_spread: float = 0.05,
) -> str:
    """Classify the diagnostic without changing any experiment gate."""
    if not reference_valid:
        return "protocol-invalid"
    if control_summary is None or not control_batch_summaries:
        return "evaluation-noise/inconclusive"

    batch_passes = [
        float(cast(float, summary["win_share"])) >= min_win_share
        and float(cast(float, summary["seat_spread"])) <= max_seat_spread
        for summary in control_batch_summaries
    ]
    if any(batch_passes) and not all(batch_passes):
        return "evaluation-noise/inconclusive"
    if (
        float(cast(float, control_summary["win_share"])) >= min_win_share
        and float(cast(float, control_summary["seat_spread"])) <= max_seat_spread
    ):
        return "protocol-valid"
    return "recipe-failure"


__all__ = [
    "classify_diagnostic",
    "config_fingerprint",
    "reference_reproduces",
    "resolve_manifest_files",
    "validate_seed_plan",
]
