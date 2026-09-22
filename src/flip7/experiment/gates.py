"""Gate configuration and evaluation logic driven by experiment YAML."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from flip7.experiment.parsing import as_mapping


def _as_float(value: object) -> float:
    return float(cast(float | int | str, value))


@dataclass(frozen=True, slots=True)
class GateConfig:
    """Configurable decision thresholds for research phase promotion."""

    min_baseline_win_share: float = 0.604
    max_seat_spread: float = 0.05
    min_heldout_difference: float = 0.03
    max_seed_degradation: float = -0.02
    min_final_rating: float = 1500.0
    min_final_minus_warmup_elo: float = 25.0
    min_behavioral_js_ratio: float = 1.20
    min_action_disagreement_delta: float = 0.05
    min_paired_final_warmup_difference: float = 0.0

    @classmethod
    def from_config(cls, root: Mapping[str, object]) -> GateConfig:
        """Parse GateConfig from root configuration mapping with safe defaults."""
        raw_gates = root.get("gates")
        if not isinstance(raw_gates, dict):
            return cls()
        gates = cast(Mapping[str, object], raw_gates)
        min_warmup = gates.get(
            "min_final_minus_warmup_elo", gates.get("min_warmup_gain", 25.0)
        )
        return cls(
            min_baseline_win_share=_as_float(
                gates.get("min_baseline_win_share", 0.604)
            ),
            max_seat_spread=_as_float(gates.get("max_seat_spread", 0.05)),
            min_heldout_difference=_as_float(gates.get("min_heldout_difference", 0.03)),
            max_seed_degradation=_as_float(gates.get("max_seed_degradation", -0.02)),
            min_final_rating=_as_float(gates.get("min_final_rating", 1500.0)),
            min_final_minus_warmup_elo=_as_float(min_warmup),
            min_behavioral_js_ratio=_as_float(
                gates.get("min_behavioral_js_ratio", 1.20)
            ),
            min_action_disagreement_delta=_as_float(
                gates.get("min_action_disagreement_delta", 0.05)
            ),
            min_paired_final_warmup_difference=_as_float(
                gates.get("min_paired_final_warmup_difference", 0.0)
            ),
        )


def evaluate_phase7_gates(
    control_rotated: Mapping[str, object],
    league_rotated: Mapping[str, object],
    league_heldout: Mapping[str, object],
    latest_heldout: Mapping[str, object],
    league_warmup_delta_min: float | None,
    league_aggregate: Mapping[str, object],
    reference_win_share: float,
    reference_spread: float,
    gates: GateConfig,
) -> dict[str, bool]:
    """Evaluate normative Phase 7 decision gates."""
    return {
        "control_reproduces_phase6": (
            abs(_as_float(control_rotated["win_share"]) - reference_win_share) <= 0.01
            and abs(_as_float(control_rotated["seat_spread"]) - reference_spread)
            <= 0.01
        ),
        "league_robustness": (
            _as_float(league_rotated["win_share"]) >= gates.min_baseline_win_share
            and _as_float(league_rotated["seat_spread"]) <= gates.max_seat_spread
        ),
        # Note: max_seed_degradation carries different semantics per phase.
        # Here it is the aggregate held-out lower bound (phase7.yaml: -0.05);
        # stability runners use it as the minimum paired per-seed difference.
        "population_protection": (
            _as_float(league_heldout["win_share"])
            >= _as_float(latest_heldout["win_share"]) + gates.max_seed_degradation
            and _as_float(league_heldout["win_share"])
            >= _as_float(latest_heldout["win_share"]) + gates.min_heldout_difference
        ),
        "tournament_adaptation": (
            league_warmup_delta_min is not None
            and float(league_warmup_delta_min) >= gates.min_final_minus_warmup_elo
            and _as_float(league_aggregate["final_rating_median"])
            >= gates.min_final_rating
        ),
    }


def is_candidate_robust(row: Mapping[str, object], gates: GateConfig) -> bool:
    """Check if a candidate row meets baseline strength and seat spread criteria."""
    baseline = as_mapping(row["baseline"], "baseline")
    return (
        _as_float(baseline["seat_spread"]) <= gates.max_seat_spread
        and _as_float(baseline["win_share"]) >= gates.min_baseline_win_share
    )
