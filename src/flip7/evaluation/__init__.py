"""Reproducible baseline-agent evaluation utilities."""

from flip7.evaluation.metrics import GameResult, MatchupMetrics
from flip7.evaluation.phase6 import (
    PHASE6_MATCHUPS,
    PHASE6_SEED_BASES,
    run_rotated_matchups,
    summarize_rotated_results,
    write_phase6_results,
)
from flip7.evaluation.tournament import run_game, run_matchup, write_results

__all__ = [
    "GameResult",
    "MatchupMetrics",
    "PHASE6_MATCHUPS",
    "PHASE6_SEED_BASES",
    "run_game",
    "run_matchup",
    "run_rotated_matchups",
    "summarize_rotated_results",
    "write_phase6_results",
    "write_results",
]
