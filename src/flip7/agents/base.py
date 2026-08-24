"""Shared interfaces and observation helpers for baseline policies."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from flip7.envs.encodings import TARGET_OFFSET

type Observation = NDArray[np.float32]
type ActionMask = NDArray[np.int8]


class Agent(Protocol):
    """Callable policy interface shared with the environment opponent API."""

    def __call__(self, observation: Observation, action_mask: ActionMask) -> int:
        """Choose one currently legal discrete action."""
        ...


type AgentFactory = Callable[[], Agent]


class ObservationView:
    """Decode the stable competitive/deck-aware observation layout."""

    def __init__(self, observation: Observation, action_mask: ActionMask) -> None:
        self.observation = observation
        self.action_mask = action_mask
        self.player_count = len(action_mask) - 2
        if self.player_count < 3:
            raise ValueError("action mask does not identify a valid player count")
        competitive_size = 30 * self.player_count + 6
        if observation.size not in {33, competitive_size, competitive_size + 24}:
            raise ValueError("unsupported observation size for a baseline agent")
        self.is_basic = observation.size == 33
        self.is_deck_aware = observation.size == competitive_size + 24

    @property
    def ego_id(self) -> int:
        if self.is_basic:
            return 0
        start = 26 * self.player_count
        ego = np.flatnonzero(self.observation[start : start + self.player_count])
        if ego.size != 1:
            raise ValueError("competitive observation must contain one ego seat")
        return int(ego[0])

    def player_features(self, player_id: int) -> NDArray[np.float32]:
        if self.is_basic:
            if player_id != 0:
                raise ValueError("basic observations contain only the ego player")
            return self.observation[:26]
        start = 26 * player_id
        return self.observation[start : start + 26]

    @property
    def ego_features(self) -> NDArray[np.float32]:
        return self.player_features(self.ego_id)

    @property
    def round_number(self) -> float:
        if self.is_basic:
            return float(self.observation[26] * 30.0)
        return float(self.observation[30 * self.player_count])

    @property
    def pending_action_index(self) -> int:
        if self.is_basic:
            return int(np.argmax(self.observation[27:31]))
        start = 30 * self.player_count + 1
        return int(np.argmax(self.observation[start : start + 4]))

    @property
    def deck_counts(self) -> NDArray[np.float32]:
        if not self.is_deck_aware:
            raise ValueError("this agent requires a deck_aware observation")
        start = 30 * self.player_count + 6
        return self.observation[start : start + 13]

    @property
    def deck_size(self) -> float:
        if not self.is_deck_aware:
            raise ValueError("this agent requires a deck_aware observation")
        return float(self.observation[30 * self.player_count + 28])

    def legal_targets(self) -> tuple[int, ...]:
        return tuple(
            int(action - TARGET_OFFSET)
            for action in np.flatnonzero(self.action_mask)
            if action >= TARGET_OFFSET
        )


def first_legal(mask: ActionMask, preferred: int) -> int:
    """Return a preferred action or the first legal action as a safe fallback."""
    if 0 <= preferred < len(mask) and mask[preferred]:
        return preferred
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        raise RuntimeError("no legal actions remain")
    return int(legal[0])


def choose_target(view: ObservationView) -> int:
    """Choose a deterministic target, prioritizing the highest-scoring threat."""
    targets = view.legal_targets()
    if not targets:
        raise RuntimeError("no legal target actions remain")
    ego_id = view.ego_id
    ranked = sorted(
        targets,
        key=lambda player_id: (
            float(view.player_features(player_id)[25]),
            float(view.player_features(player_id)[24]),
            -player_id,
        ),
        reverse=True,
    )
    pending = view.pending_action_index
    if pending == 3 and ego_id in targets:
        return TARGET_OFFSET + ego_id
    return TARGET_OFFSET + ranked[0]


def round_value(features: NDArray[np.float32]) -> float:
    """Estimate the current round value represented by player features."""
    number_value = sum(value for value, present in enumerate(features[:13]) if present)
    modifier_value = sum(
        modifier
        for modifier, present in zip((2, 4, 6, 8, 10), features[14:19], strict=True)
        if present
    )
    multiplier = 2.0 if features[19] else 1.0
    return float((number_value + modifier_value) * multiplier)


def held_number_values(features: NDArray[np.float32]) -> tuple[int, ...]:
    """Return number faces currently held by the ego player."""
    return tuple(int(value) for value, present in enumerate(features[:13]) if present)
