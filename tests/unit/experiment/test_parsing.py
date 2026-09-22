"""Unit tests for flip7.experiment.parsing."""

import pytest

from flip7.experiment.parsing import (
    as_ints,
    as_list,
    as_mapping,
    as_pairs,
    as_strings,
    mappo_config,
    training_config,
)


def test_as_mapping_valid():
    assert as_mapping({"a": 1}, "test") == {"a": 1}


def test_as_mapping_invalid():
    with pytest.raises(ValueError, match="test must be a mapping"):
        as_mapping([1, 2], "test")


def test_as_list_valid():
    assert as_list([1, 2], "test") == [1, 2]


def test_as_list_invalid():
    with pytest.raises(ValueError, match="test must be a list"):
        as_list("not a list", "test")


def test_as_ints_valid():
    assert as_ints([1, 2, 3], "test") == (1, 2, 3)


def test_as_ints_rejects_bool():
    with pytest.raises(ValueError, match="test must contain only integers"):
        as_ints([1, True, 3], "test")


def test_as_strings_valid():
    assert as_strings(["a", "b"], "test") == ("a", "b")


def test_as_strings_invalid():
    with pytest.raises(ValueError, match="test must contain only strings"):
        as_strings(["a", 1], "test")


def test_as_pairs_valid():
    assert as_pairs([["a", "b"], ["c", "d"]], "test") == (("a", "b"), ("c", "d"))


def test_as_pairs_invalid_length():
    with pytest.raises(ValueError, match="test entries must contain two names"):
        as_pairs([["a", "b", "c"]], "test")


def test_as_pairs_empty():
    with pytest.raises(ValueError, match="test must not be empty"):
        as_pairs([], "test")


def test_training_config_and_overrides():
    root = {
        "players": 3,
        "env": {
            "learner_id": 0,
            "learner_seat_mode": "random",
            "observation": "basic",
            "reward": "sparse_win",
        },
        "training": {
            "algorithm": "ppo",
            "updates": 50,
            "rollout_steps": 1024,
            "learning_rate": 0.0003,
        },
    }
    cfg = training_config(root, seed=42)
    assert cfg.seed == 42
    assert cfg.player_count == 3
    assert cfg.updates == 50
    assert cfg.observation == "basic"
    assert cfg.learner_seat_mode == "random"

    # With overrides
    cfg2 = training_config(
        root,
        seed=99,
        updates=10,
        observation_override="deck_aware",
        learner_seat_mode_override="fixed",
        training_override={"learning_rate": 0.0001},
    )
    assert cfg2.seed == 99
    assert cfg2.updates == 10
    assert cfg2.observation == "deck_aware"
    assert cfg2.learner_seat_mode == "fixed"
    assert cfg2.learning_rate == 0.0001


def test_mappo_config_generation():
    root = {
        "players": 3,
        "env": {
            "observation": "basic",
            "reward": "sparse_win",
        },
        "training": {
            "algorithm": "ppo",
            "updates": 100,
            "network": "separate",
            "critic_seat_conditioned": True,
            "learning_rate": 0.0003,
        },
    }
    cfg = mappo_config(root, seed=7, updates=20)
    assert cfg.seed == 7
    assert cfg.updates == 20
    assert cfg.player_count == 3
    assert cfg.observation == "basic"
