"""Population-based self-play training built on the Phase 5 PPO trainer."""

from __future__ import annotations

import json
import random
from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch

from flip7.agents import Agent, PPOAgent
from flip7.envs import (
    ObservationFamily,
    action_space_size,
    agent_name,
)
from flip7.envs import (
    observation_size as get_observation_size,
)
from flip7.training.ppo import (
    EpisodeLineup,
    PPOConfig,
    PPOTrainer,
    baseline_factories,
)


@dataclass(frozen=True, slots=True)
class LeagueConfig:
    """Configuration for the frozen-policy opponent population."""

    warmup_updates: int = 10
    snapshot_interval: int = 5
    max_snapshots: int = 8
    learned_opponent_probability: float = 0.5
    baseline_observation: str = "deck_aware"
    baseline_names: tuple[str, ...] = ("random", "threshold", "risk", "ev", "dp")

    def __post_init__(self) -> None:
        if self.warmup_updates < 0:
            raise ValueError("warmup_updates must be non-negative")
        if self.snapshot_interval < 1:
            raise ValueError("snapshot_interval must be positive")
        if self.max_snapshots < 1:
            raise ValueError("max_snapshots must be positive")
        if not 0.0 <= self.learned_opponent_probability <= 1.0:
            raise ValueError("learned_opponent_probability must be between 0 and 1")
        if not self.baseline_names:
            raise ValueError("at least one baseline opponent is required")
        missing = set(self.baseline_names) - set(baseline_factories())
        if missing:
            raise ValueError(f"unknown baseline opponents: {sorted(missing)}")
        ObservationFamily(self.baseline_observation)


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    """Immutable metadata describing one frozen learned policy."""

    policy_id: str
    path: Path
    update: int
    seed: int
    observation: ObservationFamily
    observation_size: int
    action_size: int
    config: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible snapshot metadata."""
        return {
            "policy_id": self.policy_id,
            "checkpoint": str(self.path),
            "update": self.update,
            "seed": self.seed,
            "observation": self.observation.value,
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "config": dict(self.config),
        }


class PolicyLeague:
    """Sample diverse frozen policies and baseline agents for each episode."""

    def __init__(
        self,
        config: LeagueConfig,
        *,
        observation: ObservationFamily,
        source_seed: int,
        player_count: int | None = None,
    ) -> None:
        if player_count is not None and player_count < 3:
            raise ValueError("player_count must be at least three")
        self.config = config
        self.observation = observation
        self.source_seed = source_seed
        self.player_count = player_count
        self._snapshots: deque[PolicySnapshot] = deque(maxlen=config.max_snapshots)
        self._exposure: Counter[str] = Counter()

    @property
    def snapshots(self) -> tuple[PolicySnapshot, ...]:
        """Return snapshots in chronological population order."""
        return tuple(self._snapshots)

    @property
    def exposure(self) -> Mapping[str, int]:
        """Return cumulative opponent-selection counts."""
        return dict(self._exposure)

    def episode_lineup(self, rng: random.Random, player_count: int) -> EpisodeLineup:
        """Build one seeded lineup with a uniformly sampled learner seat."""
        learner_id = rng.randrange(player_count)
        used_baselines: set[str] = set()
        used_snapshots: set[str] = set()
        opponents: dict[str, Agent] = {}
        opponent_observations: dict[str, ObservationFamily] = {}

        for seat in range(player_count):
            if seat == learner_id:
                continue
            kind, selected = self._select_entry(rng, used_baselines, used_snapshots)
            policy, observation, key = self._instantiate_entry(rng, kind, selected)
            opponents[agent_name(seat)] = policy
            opponent_observations[agent_name(seat)] = observation
            self._exposure[key] += 1

        return EpisodeLineup(learner_id, opponents, opponent_observations)

    def register_snapshot(
        self, path: Path, *, update: int, seed: int
    ) -> PolicySnapshot:
        """Validate and add a checkpoint to the bounded frozen population."""
        if update < 1:
            raise ValueError("snapshot update must be positive")
        payload_value: object = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload_value, dict):
            raise ValueError("snapshot checkpoint is missing its config")
        payload = cast(dict[str, object], payload_value)
        raw_config_value = payload.get("config")
        if not isinstance(raw_config_value, dict):
            raise ValueError("snapshot checkpoint is missing its config")
        raw_config = cast(dict[str, object], raw_config_value)
        observation_value = raw_config.get("observation")
        observation_size_value = raw_config.get("observation_size")
        action_size_value = raw_config.get("action_size")
        if (
            not isinstance(observation_value, str)
            or not isinstance(observation_size_value, int)
            or not isinstance(action_size_value, int)
        ):
            raise ValueError("snapshot checkpoint has invalid dimensions")
        try:
            observation = ObservationFamily(observation_value)
        except ValueError as exc:
            raise ValueError("snapshot checkpoint has invalid dimensions") from exc
        observation_size = observation_size_value
        action_size = action_size_value
        if observation is not self.observation:
            raise ValueError("snapshot observation does not match the league")
        if observation_size < 1 or action_size < 1:
            raise ValueError("snapshot dimensions must be positive")
        if self.player_count is not None:
            expected_observation_size = get_observation_size(
                self.player_count, self.observation
            )
            expected_action_size = action_space_size(self.player_count)
            if (
                observation_size != expected_observation_size
                or action_size != expected_action_size
            ):
                raise ValueError("snapshot dimensions do not match the league")

        policy_id = f"seed-{seed}-update-{update:04d}"
        snapshot = PolicySnapshot(
            policy_id=policy_id,
            path=path,
            update=update,
            seed=seed,
            observation=observation,
            observation_size=observation_size,
            action_size=action_size,
            config=raw_config,
        )
        self._snapshots.append(snapshot)
        return snapshot

    def as_dict(self) -> dict[str, object]:
        """Return population metadata and exposure counts for artifacts."""
        return {
            "config": asdict(self.config),
            "observation": self.observation.value,
            "source_seed": self.source_seed,
            "snapshots": [snapshot.as_dict() for snapshot in self._snapshots],
            "opponent_exposure": dict(sorted(self._exposure.items())),
        }

    def _select_entry(
        self,
        rng: random.Random,
        used_baselines: set[str],
        used_snapshots: set[str],
    ) -> tuple[str, str | PolicySnapshot]:
        unused_baselines = [
            name for name in self.config.baseline_names if name not in used_baselines
        ]
        baselines = unused_baselines or list(self.config.baseline_names)
        unused_snapshots = [
            snapshot
            for snapshot in self._snapshots
            if snapshot.policy_id not in used_snapshots
        ]
        snapshots = unused_snapshots or list(self._snapshots)
        choose_learned = bool(snapshots) and (
            not baselines or rng.random() < self.config.learned_opponent_probability
        )
        if choose_learned:
            selected = rng.choice(snapshots)
            used_snapshots.add(selected.policy_id)
            return "snapshot", selected
        if baselines:
            selected_baseline = rng.choice(baselines)
            used_baselines.add(selected_baseline)
            return "baseline", selected_baseline
        if snapshots:
            selected = rng.choice(snapshots)
            used_snapshots.add(selected.policy_id)
            return "snapshot", selected
        raise RuntimeError("the policy league has no available opponents")

    def _instantiate_entry(
        self,
        rng: random.Random,
        kind: str,
        selected: str | PolicySnapshot,
    ) -> tuple[Agent, ObservationFamily, str]:
        seed = rng.randrange(2**31)
        if kind == "baseline":
            if not isinstance(selected, str):
                raise TypeError("baseline league entries must be named strings")
            factory = baseline_factories(random_seed=seed)[selected]
            return (
                factory(),
                ObservationFamily(self.config.baseline_observation),
                (f"baseline:{selected}"),
            )
        if not isinstance(selected, PolicySnapshot):
            raise TypeError("learned league entries must be policy snapshots")
        policy = PPOAgent.from_checkpoint(
            selected.path,
            deterministic=False,
            seed=seed,
            device="cpu",
        )
        return policy, selected.observation, f"snapshot:{selected.policy_id}"


class LeaguePPOTrainer(PPOTrainer):
    """Train an active PPO policy against a bounded frozen policy league."""

    def __init__(
        self,
        config: PPOConfig,
        *,
        league_config: LeagueConfig | None = None,
    ) -> None:
        selected_config = LeagueConfig() if league_config is None else league_config
        self.league = PolicyLeague(
            selected_config,
            observation=ObservationFamily(config.observation),
            source_seed=config.seed,
            player_count=config.player_count,
        )
        super().__init__(
            config,
            opponent_names=selected_config.baseline_names,
            opponent_provider=self.league.episode_lineup,
        )

    def train(self, checkpoint: Path | None = None) -> list[dict[str, float]]:
        """Train and periodically archive frozen snapshots for later episodes."""
        if checkpoint is None:
            msg = "LeaguePPOTrainer requires a checkpoint path for snapshots"
            raise ValueError(msg)
        with self.worker_pool_scope():
            history: list[dict[str, float]] = []
            snapshot_dir = checkpoint.parent / "checkpoints"
            for update in range(1, self.config.updates + 1):
                rollout = self.collect_rollout()
                metrics = self.update(rollout)
                metrics["update"] = float(update)
                metrics["mean_reward"] = float(rollout.rewards.mean())

                self.save_checkpoint(checkpoint, update)
                if (
                    update >= self.league.config.warmup_updates
                    and update % self.league.config.snapshot_interval == 0
                ):
                    snapshot_path = snapshot_dir / f"update-{update:04d}.pt"
                    policy_id = f"seed-{self.config.seed}-update-{update:04d}"
                    self.save_checkpoint(
                        snapshot_path,
                        update,
                        metadata={
                            "policy_id": policy_id,
                            "source_seed": self.config.seed,
                            "snapshot_update": update,
                            "observation": self.config.observation,
                            "observation_size": self.observation_size,
                            "action_size": self.action_size,
                        },
                    )
                    self.league.register_snapshot(
                        snapshot_path, update=update, seed=self.config.seed
                    )

                total_exposure = sum(self.league.exposure.values())
                learned_exposure = sum(
                    count
                    for key, count in self.league.exposure.items()
                    if key.startswith("snapshot:")
                )
                metrics["population_size"] = float(len(self.league.snapshots))
                metrics["learned_opponent_share"] = (
                    learned_exposure / total_exposure if total_exposure else 0.0
                )
                history.append(metrics)
            return history


def write_population(path: Path, league: PolicyLeague) -> None:
    """Write deterministic population metadata as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(league.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "LeagueConfig",
    "LeaguePPOTrainer",
    "PolicyLeague",
    "PolicySnapshot",
    "write_population",
]
