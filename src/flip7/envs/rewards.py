"""Episode reward functions for Flip 7 environments."""

from __future__ import annotations

from enum import StrEnum

from flip7.core.engine import TransitionResult

SCORE_SCALE = 200.0


class RewardMode(StrEnum):
    """Named reward functions for a complete game episode."""

    SPARSE_WIN = "sparse_win"
    ROUND_SCORE = "round_score"


def rewards_from_result(
    result: TransitionResult,
    player_count: int,
    mode: RewardMode,
) -> dict[int, float]:
    """Return per-player rewards produced by one engine transition."""
    rewards = dict.fromkeys(range(player_count), 0.0)
    if mode is RewardMode.ROUND_SCORE:
        for player_id, round_score in result.score_changes:
            rewards[player_id] += round_score / SCORE_SCALE
    if result.is_game_terminal and result.state.winning_player_ids:
        share = 1.0 / len(result.state.winning_player_ids)
        for player_id in result.state.winning_player_ids:
            rewards[player_id] += share
    return rewards
