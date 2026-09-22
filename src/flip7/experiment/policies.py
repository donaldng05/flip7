"""Policy caching and tournament participant roster construction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from flip7.agents import PPOAgent
from flip7.envs import ObservationFamily
from flip7.evaluation.phase7 import TournamentParticipant
from flip7.training import baseline_factories


def cached_policy(path: Path | str) -> Callable[[], PPOAgent]:
    """Return a factory function that loads and memoizes a deterministic PPOAgent."""
    resolved_path = Path(path)
    policy_instance: PPOAgent | None = None

    def factory() -> PPOAgent:
        nonlocal policy_instance
        if policy_instance is None:
            policy_instance = PPOAgent.from_checkpoint(
                resolved_path, deterministic=True
            )
        return policy_instance

    return factory


def build_participants(
    checkpoint: Path | str,
    observation: ObservationFamily,
    snapshots: Sequence[Any] = (),
    warmup: Any | None = None,
    *,
    include_snapshots: bool = True,
) -> tuple[TournamentParticipant, ...]:
    """Construct participants from baseline roster, checkpoint, and snapshots."""
    checkpoint_path = Path(checkpoint)
    roster = baseline_factories()
    participants = [
        TournamentParticipant(name, factory, ObservationFamily.DECK_AWARE)
        for name, factory in roster.items()
    ]
    participants.append(
        TournamentParticipant("final", cached_policy(checkpoint_path), observation)
    )
    if include_snapshots:
        for snapshot in snapshots:
            participants.append(
                TournamentParticipant(
                    snapshot.policy_id,
                    cached_policy(Path(snapshot.path)),
                    snapshot.observation,
                )
            )
    if warmup is not None and all(
        participant.name != warmup.policy_id for participant in participants
    ):
        participants.append(
            TournamentParticipant(
                warmup.policy_id,
                cached_policy(Path(warmup.path)),
                warmup.observation,
            )
        )
    return tuple(participants)
