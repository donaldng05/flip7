"""Tests for baseline metric aggregation and seeded tournament execution."""

from collections.abc import Mapping
from typing import cast

from flip7.agents import FixedThresholdAgent, RandomLegalAgent
from flip7.envs import ObservationFamily
from flip7.evaluation import (
    PHASE6_MATCHUPS,
    GameResult,
    MatchupMetrics,
    run_matchup,
    run_rotated_matchups,
    summarize_rotated_results,
)
from flip7.training import baseline_factories


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


def test_rotated_matchups_cover_every_learner_seat_reproducibly() -> None:
    def learner() -> RandomLegalAgent:
        return RandomLegalAgent(seed=31)

    first = run_rotated_matchups(
        learner,
        baseline_factories(),
        games=1,
        seed_bases=(100, 200, 300),
        observation=ObservationFamily.BASIC,
    )
    second = run_rotated_matchups(
        learner,
        baseline_factories(),
        games=1,
        seed_bases=(100, 200, 300),
        observation=ObservationFamily.BASIC,
    )

    assert len(first) == len(PHASE6_MATCHUPS) * 3
    assert [result.as_dict() for result in first] == [
        result.as_dict() for result in second
    ]
    assert [result.agents.index("ppo") for result in first] == [
        0,
        1,
        2,
        0,
        1,
        2,
        0,
        1,
        2,
    ]
    summary = summarize_rotated_results(first)
    per_seat = cast(Mapping[str, object], summary["per_seat"])
    assert set(per_seat) == {"0", "1", "2"}
    per_matchup = cast(Mapping[str, object], summary["per_matchup"])
    assert set(per_matchup) == {
        "random/threshold",
        "risk/ev",
        "dp/threshold",
    }
    seat_spread = cast(float, summary["seat_spread"])
    assert 0.0 <= seat_spread <= 1.0
