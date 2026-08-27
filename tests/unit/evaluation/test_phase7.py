from flip7.agents import FixedThresholdAgent, RandomLegalAgent
from flip7.envs import ObservationFamily
from flip7.evaluation import (
    EloTable,
    TournamentParticipant,
    run_round_robin,
)


def test_elo_updates_preserve_total_rating_and_handle_ties() -> None:
    table = EloTable(("a", "b", "c"))
    before = sum(table.ratings.values())

    table.update(("a", "b", "c"), (0,))
    table.update(("a", "b", "c"), (0, 1))

    assert sum(table.ratings.values()) == before
    assert table.games == {"a": 2, "b": 2, "c": 2}
    assert table.win_shares["a"] == 1.5
    assert table.win_shares["b"] == 0.5


def _participants() -> tuple[TournamentParticipant, ...]:
    return (
        TournamentParticipant(
            "random", lambda: RandomLegalAgent(seed=1), ObservationFamily.DECK_AWARE
        ),
        TournamentParticipant(
            "threshold",
            lambda: FixedThresholdAgent(threshold=15),
            ObservationFamily.DECK_AWARE,
        ),
        TournamentParticipant(
            "random2", lambda: RandomLegalAgent(seed=2), ObservationFamily.DECK_AWARE
        ),
    )


def test_round_robin_is_seeded_and_rotates_all_six_seat_orders() -> None:
    first = run_round_robin(_participants(), games=1, seed=77)
    second = run_round_robin(_participants(), games=1, seed=77)

    assert first.games == 6
    assert first.lineup_games == 6
    assert first.as_dict() == second.as_dict()
    assert len(first.matchups) == 1
    assert len(first.elo["ratings"]) == 3  # type: ignore[index]
