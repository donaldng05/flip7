"""Unit tests for flip7.experiment.policies."""

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

from flip7.agents import Agent
from flip7.envs import ObservationFamily
from flip7.experiment.policies import build_participants, cached_policy
from flip7.training import PolicySnapshot


def test_cached_policy(tmp_path: Path) -> None:
    ckpt: Path = tmp_path / "model.pt"
    ckpt.write_bytes(b"data")

    with patch("flip7.agents.PPOAgent.from_checkpoint") as mock_from_ckpt:
        mock_agent = cast(Agent, MagicMock(spec=Agent))
        mock_from_ckpt.return_value = mock_agent

        factory = cached_policy(ckpt)
        agent1 = factory()
        agent2 = factory()
        assert agent1 is mock_agent
        assert agent2 is mock_agent
        assert mock_from_ckpt.call_count == 1


def test_build_participants(tmp_path: Path) -> None:
    ckpt: Path = tmp_path / "model.pt"
    ckpt.write_bytes(b"data")

    snap_file: Path = tmp_path / "snap1.pt"
    snap_file.write_bytes(b"snap")
    snapshot = PolicySnapshot(
        policy_id="snap_1",
        path=snap_file,
        update=10,
        seed=1,
        observation=ObservationFamily.BASIC,
        observation_size=10,
        action_size=2,
        config={},
    )

    warmup_file: Path = tmp_path / "warmup.pt"
    warmup_file.write_bytes(b"warmup")
    warmup = PolicySnapshot(
        policy_id="warmup_1",
        path=warmup_file,
        update=5,
        seed=1,
        observation=ObservationFamily.BASIC,
        observation_size=10,
        action_size=2,
        config={},
    )

    with patch("flip7.agents.PPOAgent.from_checkpoint"):
        participants = build_participants(
            checkpoint=ckpt,
            observation=ObservationFamily.BASIC,
            snapshots=[snapshot],
            warmup=warmup,
            include_snapshots=True,
        )
        names = [p.name for p in participants]
        assert "final" in names
        assert "snap_1" in names
        assert "warmup_1" in names
        # baselines are also included
        assert "random" in names
