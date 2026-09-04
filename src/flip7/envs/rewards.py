"""Episode reward functions for Flip 7 environments."""

from __future__ import annotations

import math
from enum import StrEnum

from flip7.core.engine import TransitionResult
from flip7.core.state import GameState

SCORE_SCALE = 200.0


class RewardMode(StrEnum):
    """Named reward functions for a complete game episode."""

    SPARSE_WIN = "sparse_win"
    ROUND_SCORE = "round_score"
    POTENTIAL_WIN = "potential_win"


def rewards_from_result(
    result: TransitionResult,
    player_count: int,
    mode: RewardMode,
    *,
    previous_state: GameState | None = None,
    potential_discount: float = 0.99,
) -> dict[int, float]:
    """Return per-player rewards produced by one engine transition.

    ``potential_win`` adds a public score-differential potential delta to the
    terminal sparse-win reward.  The optional previous state keeps the existing
    reward helper compatible with sparse and round-score modes.
    """
    rewards = dict.fromkeys(range(player_count), 0.0)
    if mode is RewardMode.ROUND_SCORE:
        for player_id, round_score in result.score_changes:
            rewards[player_id] += round_score / SCORE_SCALE
    elif mode is RewardMode.POTENTIAL_WIN:
        if previous_state is None:
            raise ValueError("potential_win requires the previous game state")
        if not 0.0 < potential_discount <= 1.0 or not math.isfinite(potential_discount):
            raise ValueError("potential_discount must be in (0, 1]")
        for player_id in range(player_count):
            before = score_differential_potential(previous_state, player_id)
            after = score_differential_potential(result.state, player_id)
            rewards[player_id] += potential_discount * after - before
    if result.is_game_terminal and result.state.winning_player_ids:
        share = 1.0 / len(result.state.winning_player_ids)
        for player_id in result.state.winning_player_ids:
            rewards[player_id] += share
    return rewards


def score_differential_potential(state: GameState, player_id: int) -> float:
    """Return a bounded public score-differential potential for one player."""
    if not 0 <= player_id < len(state.players):
        raise ValueError("player_id must reference a seated player")
    own_score = state.players[player_id].cumulative_score
    opponents = [
        player.cumulative_score
        for index, player in enumerate(state.players)
        if index != player_id
    ]
    opponent_mean = sum(opponents) / max(1, len(opponents))
    differential = (own_score - opponent_mean) / SCORE_SCALE
    return max(-1.0, min(1.0, differential))


__all__ = ["RewardMode", "rewards_from_result", "score_differential_potential"]
