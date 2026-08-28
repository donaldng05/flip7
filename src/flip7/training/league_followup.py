"""Follow-up league training with behaviour-aware snapshot retention."""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from flip7.agents import Agent, PPOAgent
from flip7.envs import ObservationFamily, action_space_size, agent_name
from flip7.envs import observation_size as get_observation_size
from flip7.evaluation.diversity import (
    PolicyBehavior,
    StateBank,
    build_state_bank,
    mean_jensen_shannon_divergence,
    policy_behavior,
)
from flip7.training.league import PolicySnapshot, baseline_factories
from flip7.training.ppo import EpisodeLineup, PPOConfig, PPOTrainer


@dataclass(frozen=True, slots=True)
class FollowUpLeagueConfig:
    """Configuration for the screening and confirmation leagues."""

    warmup_updates: int = 10
    archive_interval: int = 2
    max_population: int = 12
    learned_opponent_probability: float = 0.75
    baseline_observation: str = "deck_aware"
    baseline_names: tuple[str, ...] = ("random", "threshold", "risk", "ev", "dp")
    retention_strategy: str = "novelty"
    state_bank_size: int = 2_048
    state_bank_seed: int = 70_000

    def __post_init__(self) -> None:
        if self.warmup_updates < 0:
            raise ValueError("warmup_updates must be non-negative")
        if self.archive_interval < 1:
            raise ValueError("archive_interval must be positive")
        if self.max_population < 1:
            raise ValueError("max_population must be positive")
        if not 0.0 <= self.learned_opponent_probability <= 1.0:
            raise ValueError("learned_opponent_probability must be between 0 and 1")
        if self.retention_strategy not in {"latest", "temporal", "novelty"}:
            raise ValueError("retention_strategy must be latest, temporal, or novelty")
        if self.state_bank_size < 1:
            raise ValueError("state_bank_size must be positive")
        if not self.baseline_names:
            raise ValueError("at least one baseline opponent is required")
        missing = set(self.baseline_names) - set(baseline_factories())
        if missing:
            raise ValueError(f"unknown baseline opponents: {sorted(missing)}")
        ObservationFamily(self.baseline_observation)


class DiversePolicyLeague:
    """Archive all snapshots and expose a deterministic active population."""

    def __init__(
        self,
        config: FollowUpLeagueConfig,
        *,
        observation: ObservationFamily,
        source_seed: int,
        player_count: int,
        state_bank: StateBank | None = None,
    ) -> None:
        if player_count < 3:
            raise ValueError("player_count must be at least three")
        self.config = config
        self.observation = observation
        self.source_seed = source_seed
        self.player_count = player_count
        self.state_bank = state_bank or build_state_bank(
            states=config.state_bank_size,
            seed=config.state_bank_seed,
            player_count=player_count,
            observation=observation,
        )
        if self.state_bank.observation is not observation:
            raise ValueError("state bank observation does not match the league")
        self._archives: list[PolicySnapshot] = []
        self._behaviors: dict[str, PolicyBehavior] = {}
        self._active_ids: tuple[str, ...] = ()
        self._warmup_anchor: PolicySnapshot | None = None
        self._exposure: Counter[str] = Counter()
        self._policy_cache: dict[str, PPOAgent] = {}

    @property
    def snapshots(self) -> tuple[PolicySnapshot, ...]:
        """Return the current active population in archive order."""
        active = set(self._active_ids)
        return tuple(
            snapshot for snapshot in self._archives if snapshot.policy_id in active
        )

    @property
    def archived_snapshots(self) -> tuple[PolicySnapshot, ...]:
        """Return every checkpoint archived, including evicted snapshots."""
        return tuple(self._archives)

    @property
    def warmup_anchor(self) -> PolicySnapshot | None:
        """Return the immutable update-10 (or configured warmup) anchor."""
        return self._warmup_anchor

    @property
    def exposure(self) -> Mapping[str, int]:
        return dict(self._exposure)

    def episode_lineup(self, rng: random.Random, player_count: int) -> EpisodeLineup:
        """Sample a seeded learner seat and independent no-replacement opponents."""
        if player_count != self.player_count:
            raise ValueError("episode player count does not match the league")
        learner_id = rng.randrange(player_count)
        used_baselines: set[str] = set()
        used_snapshots: set[str] = set()
        opponents: dict[str, Agent] = {}
        observations: dict[str, ObservationFamily] = {}
        for seat in range(player_count):
            if seat == learner_id:
                continue
            kind, selected = self._select_entry(rng, used_baselines, used_snapshots)
            policy, family, key = self._instantiate_entry(rng, kind, selected)
            opponents[agent_name(seat)] = policy
            observations[agent_name(seat)] = family
            self._exposure[key] += 1
        return EpisodeLineup(learner_id, opponents, observations)

    def register_snapshot(
        self, path: Path, *, update: int, seed: int
    ) -> PolicySnapshot:
        """Validate, archive, and score one frozen checkpoint."""
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
            snapshot_observation = ObservationFamily(observation_value)
        except ValueError as exc:
            raise ValueError("snapshot checkpoint has invalid dimensions") from exc
        expected_observation_size = get_observation_size(
            self.player_count, self.observation
        )
        expected_action_size = action_space_size(self.player_count)
        if (
            snapshot_observation is not self.observation
            or observation_size_value != expected_observation_size
            or action_size_value != expected_action_size
        ):
            raise ValueError("snapshot dimensions do not match the league")
        snapshot = PolicySnapshot(
            policy_id=f"seed-{seed}-update-{update:04d}",
            path=path,
            update=update,
            seed=seed,
            observation=snapshot_observation,
            observation_size=observation_size_value,
            action_size=action_size_value,
            config=raw_config,
        )
        if any(item.policy_id == snapshot.policy_id for item in self._archives):
            raise ValueError("snapshot policy_id is already archived")
        behavior = policy_behavior(snapshot, self.state_bank)
        self._archives.append(snapshot)
        self._behaviors[snapshot.policy_id] = behavior
        if update == self.config.warmup_updates and self._warmup_anchor is None:
            self._warmup_anchor = snapshot
        self._active_ids = self._retained_ids()
        return snapshot

    def behavior_signature(self, policy_id: str) -> np.ndarray:
        """Return the flattened fixed-bank probability signature."""
        behavior = self._behaviors[policy_id]
        return np.asarray(behavior.probabilities, dtype=np.float32).reshape(-1)

    def _behavior_divergence(self, first_id: str, second_id: str) -> float:
        """Return mean per-state JS divergence for two behavior signatures."""
        first = self._behaviors[first_id].probabilities
        second = self._behaviors[second_id].probabilities
        return mean_jensen_shannon_divergence(first, second)

    def as_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "observation": self.observation.value,
            "source_seed": self.source_seed,
            "state_bank": self.state_bank.as_dict(),
            "snapshots": [snapshot.as_dict() for snapshot in self.snapshots],
            "archived_snapshots": [snapshot.as_dict() for snapshot in self._archives],
            "warmup_anchor": self._warmup_anchor.as_dict()
            if self._warmup_anchor is not None
            else None,
            "retention_strategy": self.config.retention_strategy,
            "opponent_exposure": dict(sorted(self._exposure.items())),
            "behavior_signatures": {
                snapshot.policy_id: {
                    "update": snapshot.update,
                    "mean_action_entropy": self._behaviors[
                        snapshot.policy_id
                    ].mean_entropy,
                    "mean_action_support": self._behaviors[
                        snapshot.policy_id
                    ].mean_support,
                }
                for snapshot in self._archives
            },
        }

    def _retained_ids(self) -> tuple[str, ...]:
        if not self._archives:
            return ()
        limit = min(self.config.max_population, len(self._archives))
        if self.config.retention_strategy == "latest":
            return tuple(item.policy_id for item in self._archives[-limit:])
        if self.config.retention_strategy == "temporal":
            indices = np.linspace(0, len(self._archives) - 1, limit, dtype=int)
            return tuple(self._archives[int(index)].policy_id for index in indices)

        required: list[PolicySnapshot] = []
        if self._warmup_anchor is not None:
            required.append(self._warmup_anchor)
        required.append(self._archives[0])
        required.append(self._archives[len(self._archives) // 2])
        required.append(self._archives[-1])
        unique_required: list[PolicySnapshot] = []
        seen: set[str] = set()
        for item in required:
            if item.policy_id not in seen:
                unique_required.append(item)
                seen.add(item.policy_id)
        selected = unique_required[:limit]
        remaining = [item for item in self._archives if item.policy_id not in seen]
        while len(selected) < limit and remaining:
            best = max(
                remaining,
                key=lambda candidate: (
                    min(
                        self._behavior_divergence(candidate.policy_id, chosen.policy_id)
                        for chosen in selected
                    )
                    if selected
                    else 0.0,
                    candidate.update,
                    candidate.policy_id,
                ),
            )
            selected.append(best)
            remaining.remove(best)
        return tuple(item.policy_id for item in selected)

    def _select_entry(
        self,
        rng: random.Random,
        used_baselines: set[str],
        used_snapshots: set[str],
    ) -> tuple[str, str | PolicySnapshot]:
        baselines = [
            name for name in self.config.baseline_names if name not in used_baselines
        ]
        baselines = baselines or list(self.config.baseline_names)
        snapshots = [
            snapshot
            for snapshot in self.snapshots
            if snapshot.policy_id not in used_snapshots
        ]
        snapshots = snapshots or list(self.snapshots)
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
        policy_seed = rng.randrange(2**31)
        if kind == "baseline":
            if not isinstance(selected, str):
                raise TypeError("baseline entries must be named strings")
            factory = baseline_factories(random_seed=policy_seed)[selected]
            return (
                factory(),
                ObservationFamily(self.config.baseline_observation),
                f"baseline:{selected}",
            )
        if not isinstance(selected, PolicySnapshot):
            raise TypeError("snapshot entries must contain policy snapshots")
        policy = self._policy_cache.get(selected.policy_id)
        if policy is None:
            policy = PPOAgent.from_checkpoint(
                selected.path, deterministic=False, seed=policy_seed, device="cpu"
            )
            self._policy_cache[selected.policy_id] = policy
        else:
            policy.reseed(policy_seed)
        return policy, selected.observation, f"snapshot:{selected.policy_id}"


class DiverseLeaguePPOTrainer(PPOTrainer):
    """PPO trainer using behaviour-aware frozen-policy retention."""

    def __init__(
        self,
        config: PPOConfig,
        *,
        league_config: FollowUpLeagueConfig | None = None,
        state_bank: StateBank | None = None,
    ) -> None:
        selected = league_config or FollowUpLeagueConfig()
        self.league = DiversePolicyLeague(
            selected,
            observation=ObservationFamily(config.observation),
            source_seed=config.seed,
            player_count=config.player_count,
            state_bank=state_bank,
        )
        super().__init__(
            config,
            opponent_names=selected.baseline_names,
            opponent_provider=self.league.episode_lineup,
        )

    def train(self, checkpoint: Path | None = None) -> list[dict[str, float]]:
        if checkpoint is None:
            raise ValueError("DiverseLeaguePPOTrainer requires a checkpoint path")
        history: list[dict[str, float]] = []
        snapshot_dir = checkpoint.parent / "checkpoints"
        for update in range(1, self.config.updates + 1):
            rollout = self.collect_rollout()
            metrics = self.update(rollout)
            metrics["update"] = float(update)
            metrics["mean_reward"] = float(rollout.rewards.mean())
            self.save_checkpoint(checkpoint, update)
            if update >= self.league.config.warmup_updates and (
                update == self.league.config.warmup_updates
                or (update - self.league.config.warmup_updates)
                % self.league.config.archive_interval
                == 0
            ):
                snapshot_path = snapshot_dir / f"update-{update:04d}.pt"
                self.save_checkpoint(
                    snapshot_path,
                    update,
                    metadata={
                        "policy_id": f"seed-{self.config.seed}-update-{update:04d}",
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
            total = sum(self.league.exposure.values())
            learned = sum(
                count
                for key, count in self.league.exposure.items()
                if key.startswith("snapshot:")
            )
            metrics["population_size"] = float(len(self.league.snapshots))
            metrics["archived_population_size"] = float(
                len(self.league.archived_snapshots)
            )
            metrics["learned_opponent_share"] = learned / total if total else 0.0
            history.append(metrics)
        return history


def write_followup_population(path: Path, league: DiversePolicyLeague) -> None:
    """Write population and retention metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(league.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


__all__ = [
    "DiverseLeaguePPOTrainer",
    "DiversePolicyLeague",
    "FollowUpLeagueConfig",
    "write_followup_population",
]
