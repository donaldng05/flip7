"""Interpretable baseline policies for Flip 7."""

from __future__ import annotations

import random
from functools import lru_cache

import numpy as np

from flip7.agents.base import (
    ActionMask,
    Observation,
    ObservationView,
    choose_target,
    first_legal,
    held_number_values,
    round_value,
)
from flip7.envs.encodings import HIT_INDEX, STAY_INDEX


class RandomLegalAgent:
    """Uniformly sample from the action mask using an isolated RNG."""

    def __init__(
        self, rng: random.Random | None = None, *, seed: int | None = None
    ) -> None:
        if rng is not None and seed is not None:
            raise ValueError("provide rng or seed, not both")
        self._rng = rng or random.Random(seed)

    def __call__(self, observation: Observation, action_mask: ActionMask) -> int:
        del observation
        legal = np.flatnonzero(action_mask)
        if legal.size == 0:
            raise RuntimeError("no legal actions remain")
        return int(self._rng.choice(legal))


class FixedThresholdAgent:
    """Stay once estimated round value reaches the configured threshold."""

    def __init__(self, threshold: float = 15.0) -> None:
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.threshold = threshold

    def __call__(self, observation: Observation, action_mask: ActionMask) -> int:
        view = ObservationView(observation, action_mask)
        if view.legal_targets():
            return choose_target(view)
        action = (
            STAY_INDEX
            if round_value(view.ego_features) >= self.threshold
            else HIT_INDEX
        )
        return first_legal(action_mask, action)


class BustProbabilityAgent:
    """Stay when the exact visible-deck bust probability exceeds tolerance."""

    def __init__(self, risk_tolerance: float = 0.20, *, min_value: float = 0.0) -> None:
        if not 0.0 <= risk_tolerance <= 1.0:
            raise ValueError("risk_tolerance must be between 0 and 1")
        self.risk_tolerance = risk_tolerance
        self.min_value = min_value

    def __call__(self, observation: Observation, action_mask: ActionMask) -> int:
        view = ObservationView(observation, action_mask)
        if not view.is_deck_aware:
            raise ValueError("BustProbabilityAgent requires deck_aware observations")
        if view.legal_targets():
            return choose_target(view)
        counts = view.deck_counts
        total = float(counts.sum())
        bust_cards = sum(
            counts[value] for value in held_number_values(view.ego_features)
        )
        bust_probability = float(bust_cards / total) if total else 0.0
        action = (
            STAY_INDEX
            if bust_probability >= self.risk_tolerance
            and round_value(view.ego_features) >= self.min_value
            else HIT_INDEX
        )
        return first_legal(action_mask, action)


class ExpectedValueAgent:
    """Choose hit or stay using the next-card expected round value."""

    def __init__(self, *, bust_penalty: float = 0.0, win_pressure: float = 0.0) -> None:
        self.bust_penalty = bust_penalty
        self.win_pressure = win_pressure

    def __call__(self, observation: Observation, action_mask: ActionMask) -> int:
        view = ObservationView(observation, action_mask)
        if not view.is_deck_aware:
            raise ValueError("ExpectedValueAgent requires deck_aware observations")
        if view.legal_targets():
            return choose_target(view)
        counts = view.deck_counts
        total = float(counts.sum())
        held = set(held_number_values(view.ego_features))
        current = round_value(view.ego_features)
        expected_gain = 0.0
        for value, count in enumerate(counts):
            if count <= 0:
                continue
            probability = float(count / total) if total else 0.0
            expected_gain += probability * (
                -self.bust_penalty if value in held else value
            )
        score_pressure = float(view.ego_features[25]) * self.win_pressure
        action = STAY_INDEX if current >= expected_gain + score_pressure else HIT_INDEX
        return first_legal(action_mask, action)


class RoundDPAgent:
    """Solve a bounded number-card round model with memoized recursion."""

    def __init__(self, *, max_unique_numbers: int = 7) -> None:
        if not 1 <= max_unique_numbers <= 13:
            raise ValueError("max_unique_numbers must be between 1 and 13")
        self.max_unique_numbers = max_unique_numbers

    def __call__(self, observation: Observation, action_mask: ActionMask) -> int:
        view = ObservationView(observation, action_mask)
        if not view.is_deck_aware:
            raise ValueError("RoundDPAgent requires deck_aware observations")
        if view.legal_targets():
            return choose_target(view)
        counts = tuple(int(count) for count in view.deck_counts)
        held = tuple(sorted(held_number_values(view.ego_features)))
        current = round_value(view.ego_features)
        hit_value = _continuation_value(counts, held, self.max_unique_numbers)
        action = STAY_INDEX if current >= hit_value else HIT_INDEX
        return first_legal(action_mask, action)


@lru_cache(maxsize=32_768)
def _continuation_value(
    counts: tuple[int, ...], held: tuple[int, ...], max_unique_numbers: int
) -> float:
    """Expected future number value, stopping when the modeled hand is full."""
    total = sum(counts)
    if not total or len(held) >= max_unique_numbers:
        return float(sum(held))
    held_set = set(held)
    expected = 0.0
    for value, count in enumerate(counts):
        if not count:
            continue
        next_counts = list(counts)
        next_counts[value] -= 1
        if value in held_set:
            outcome = 0.0
        else:
            next_held = tuple(sorted((*held, value)))
            outcome = _continuation_value(
                tuple(next_counts), next_held, max_unique_numbers
            )
        expected += count / total * outcome
    return expected


__all__ = [
    "BustProbabilityAgent",
    "ExpectedValueAgent",
    "FixedThresholdAgent",
    "RandomLegalAgent",
    "RoundDPAgent",
]
