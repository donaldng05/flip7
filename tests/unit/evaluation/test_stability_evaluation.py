"""Focused tests for paired stability evaluation."""

from typing import cast

from flip7.agents import FixedThresholdAgent, RandomLegalAgent
from flip7.envs import ObservationFamily
from flip7.evaluation import run_paired_rotated_evaluation


def test_paired_rotated_evaluation_reuses_seed_blocks_and_all_seats() -> None:
    result = run_paired_rotated_evaluation(
        lambda: RandomLegalAgent(seed=3),
        lambda: RandomLegalAgent(seed=5),
        {
            "random": lambda: RandomLegalAgent(seed=4),
            "threshold": lambda: FixedThresholdAgent(threshold=15.0),
        },
        games=1,
        seed_bases=(12000,),
        observation=ObservationFamily.BASIC,
        matchups=(("random", "threshold"),),
    )

    assert len(result.games) == 3
    assert {game.learner_seat for game in result.games} == {0, 1, 2}
    assert len({game.seed for game in result.games}) == 3
    payload = result.as_dict()
    assert cast(int, payload["games"]) == 3
    assert len(cast(list[float], payload["game_level_95_ci"])) == 2
