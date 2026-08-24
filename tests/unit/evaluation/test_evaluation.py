"""Tests for baseline metric aggregation and seeded tournament execution."""

from flip7.agents import FixedThresholdAgent, RandomLegalAgent
from flip7.evaluation import GameResult, MatchupMetrics, run_matchup


def test_matchup_metrics_split_tied_win_share() -> None:
    metrics = MatchupMetrics(("a", "b", "c"), 3)
    metrics.add(
        GameResult(
            agents=("a", "b", "c"),
            player_count=3,
            winning_player_ids=(0, 1),
            final_scores=(200, 200, 150),
            round_scores=(10, 10, 5),
            rounds=4,
            busted_rounds=(1, 0, 0),
            flip7_events=1,
            action_counts=(
                {"hit": 3, "stay": 2, "target": 0},
                {"hit": 0, "stay": 0, "target": 0},
                {"hit": 0, "stay": 0, "target": 0},
            ),
        )
    )
    result = metrics.as_dict()
    assert result["win_share"] == {"a": 0.5, "b": 0.5, "c": 0.0}
    assert result["tie_frequency"] == 1.0
    assert result["flip7_frequency"] == 1.0


def test_seeded_smoke_matchup_is_reproducible() -> None:
    roster = {
        "random": lambda: RandomLegalAgent(seed=11),
        "threshold": lambda: FixedThresholdAgent(threshold=12),
    }
    first = run_matchup(("random", "threshold", "random"), roster, games=1, seed=19)
    second = run_matchup(("random", "threshold", "random"), roster, games=1, seed=19)
    assert first.as_dict() == second.as_dict()
    assert first.games == 1
