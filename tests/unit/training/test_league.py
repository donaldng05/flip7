import random
from pathlib import Path

import numpy as np
import pytest

from flip7.envs import ObservationFamily
from flip7.training import LeagueConfig, LeaguePPOTrainer, PolicyLeague, PPOConfig


def _small_config() -> PPOConfig:
    return PPOConfig(
        seed=41,
        observation="basic",
        learner_seat_mode="random",
        rollout_steps=8,
        updates=3,
        epochs=1,
        minibatch_size=4,
        hidden_size=16,
    )


def test_league_config_validates_population_controls() -> None:
    with pytest.raises(ValueError, match="learned_opponent_probability"):
        LeagueConfig(learned_opponent_probability=1.1)
    with pytest.raises(ValueError, match="snapshot_interval"):
        LeagueConfig(snapshot_interval=0)
    with pytest.raises(ValueError, match="unknown baseline"):
        LeagueConfig(baseline_names=("unknown",))


def test_league_training_archives_frozen_snapshots_and_tracks_exposure(
    tmp_path: Path,
) -> None:
    trainer = LeaguePPOTrainer(
        _small_config(),
        league_config=LeagueConfig(
            warmup_updates=1,
            snapshot_interval=1,
            max_snapshots=2,
            learned_opponent_probability=1.0,
            baseline_names=("random", "threshold"),
        ),
    )
    checkpoint = tmp_path / "checkpoint.pt"

    history = trainer.train(checkpoint)

    assert len(history) == 3
    assert checkpoint.exists()
    assert [snapshot.update for snapshot in trainer.league.snapshots] == [2, 3]
    assert all(snapshot.path.exists() for snapshot in trainer.league.snapshots)
    assert any(key.startswith("snapshot:") for key in trainer.league.exposure)
    assert any(key.startswith("baseline:") for key in trainer.league.exposure)
    assert all(np.isfinite(value) for value in history[-1].values())


def test_league_sampling_is_seeded_and_covers_learner_seats() -> None:
    config = LeagueConfig(baseline_names=("random", "threshold"))
    first = PolicyLeague(config, observation=ObservationFamily.BASIC, source_seed=7)
    second = PolicyLeague(config, observation=ObservationFamily.BASIC, source_seed=7)

    first_rng = random.Random(11)
    second_rng = random.Random(11)
    first_ids = [first.episode_lineup(first_rng, 3).learner_id for _ in range(32)]
    second_ids = [second.episode_lineup(second_rng, 3).learner_id for _ in range(32)]

    assert first_ids == second_ids
    assert set(first_ids) <= {0, 1, 2}
    assert set(first_ids) == {0, 1, 2}
    assert sum(first.exposure.values()) == 64


def test_snapshot_observation_must_match_league(tmp_path: Path) -> None:
    from flip7.training import PPOTrainer

    checkpoint = tmp_path / "basic.pt"
    trainer = PPOTrainer(
        PPOConfig(
            seed=5,
            observation="basic",
            rollout_steps=4,
            updates=1,
            epochs=1,
            minibatch_size=2,
            hidden_size=8,
        ),
        opponent_names=("random",),
    )
    trainer.train(checkpoint)
    league = PolicyLeague(
        LeagueConfig(baseline_names=("random",)),
        observation=ObservationFamily.COMPETITIVE,
        source_seed=5,
    )

    with pytest.raises(ValueError, match="observation"):
        league.register_snapshot(checkpoint, update=1, seed=5)


def test_snapshot_dimensions_must_match_league_player_count(tmp_path: Path) -> None:
    from flip7.training import PPOTrainer

    checkpoint = tmp_path / "four_player.pt"
    trainer = PPOTrainer(
        PPOConfig(
            seed=6,
            player_count=4,
            observation="competitive",
            rollout_steps=4,
            updates=1,
            epochs=1,
            minibatch_size=2,
            hidden_size=8,
        ),
        opponent_names=("random",),
    )
    trainer.train(checkpoint)
    league = PolicyLeague(
        LeagueConfig(baseline_names=("random",)),
        observation=ObservationFamily.COMPETITIVE,
        source_seed=6,
        player_count=3,
    )

    with pytest.raises(ValueError, match="dimensions"):
        league.register_snapshot(checkpoint, update=1, seed=6)


def test_frozen_snapshot_policy_does_not_accumulate_gradients(tmp_path: Path) -> None:
    from flip7.agents import PPOAgent

    checkpoint = tmp_path / "policy.pt"
    trainer = LeaguePPOTrainer(
        PPOConfig(
            seed=9,
            observation="basic",
            rollout_steps=4,
            updates=1,
            epochs=1,
            minibatch_size=2,
            hidden_size=8,
        ),
        league_config=LeagueConfig(baseline_names=("random",)),
    )
    trainer.train(checkpoint)
    policy = PPOAgent.from_checkpoint(checkpoint, deterministic=False, seed=3)
    observation, info = trainer.new_env().reset(seed=4)

    action = policy(observation, info["action_mask"])

    assert info["action_mask"][action] == 1
    assert all(parameter.grad is None for parameter in policy.network.parameters())


def test_frozen_snapshot_policy_can_be_reseeded_per_episode() -> None:
    from flip7.agents import ActorCritic, PPOAgent

    policy = PPOAgent(ActorCritic(33, 5, 8), deterministic=False, seed=1)
    observation = np.zeros(33, dtype=np.float32)
    mask = np.ones(5, dtype=np.int8)

    policy.reseed(12)
    first = [policy(observation, mask) for _ in range(4)]
    policy.reseed(12)
    second = [policy(observation, mask) for _ in range(4)]

    assert first == second
