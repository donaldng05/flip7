"""A small, reproducible PPO trainer for one learner against frozen opponents."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, optim
from torch.distributions import Categorical
from torch.nn import functional as F

from flip7 import __version__
from flip7.agents import (
    ActorCritic,
    Agent,
    AgentFactory,
    BustProbabilityAgent,
    ExpectedValueAgent,
    FixedThresholdAgent,
    RandomLegalAgent,
    RoundDPAgent,
    SeparateActorCritic,
    seed_torch,
)
from flip7.agents.learned import masked_logits
from flip7.envs import Flip7VsOpponentsEnv, ObservationFamily, RewardMode, agent_name


@dataclass(frozen=True)
class PPOConfig:
    """Configuration for a bounded PPO experiment."""

    seed: int = 7
    player_count: int = 3
    learner_id: int = 0
    learner_seat_mode: str = "fixed"
    observation: str = "deck_aware"
    reward: str = "sparse_win"
    hidden_size: int = 128
    rollout_steps: int = 256
    updates: int = 10
    epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    device: str = "cpu"
    network: str = "shared"
    critic_seat_conditioned: bool = False
    learning_rate_end: float | None = None
    entropy_coefficient_end: float | None = None
    target_kl: float | None = None
    value_clip_epsilon: float | None = None


@dataclass(frozen=True)
class Rollout:
    observations: NDArray[np.float32]
    masks: NDArray[np.int8]
    actions: NDArray[np.int64]
    old_log_probs: NDArray[np.float32]
    rewards: NDArray[np.float32]
    dones: NDArray[np.bool_]
    values: NDArray[np.float32]
    next_value: float
    seat_ids: NDArray[np.int64] | None = None
    segment_ends: NDArray[np.bool_] | None = None
    segment_next_values: NDArray[np.float32] | None = None


@dataclass(frozen=True, slots=True)
class EpisodeLineup:
    """Policies and observation families for one learner episode."""

    learner_id: int
    opponents: Mapping[str, Agent]
    opponent_observations: Mapping[str, ObservationFamily]


type OpponentProvider = Callable[[random.Random, int], EpisodeLineup]
type SeatOpponentProvider = Callable[[random.Random, int, int], EpisodeLineup]


def compute_gae(
    rewards: NDArray[np.float32],
    dones: NDArray[np.bool_],
    values: NDArray[np.float32],
    next_value: float,
    gamma: float,
    gae_lambda: float,
    *,
    segment_ends: NDArray[np.bool_] | None = None,
    segment_next_values: NDArray[np.float32] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute terminal-aware generalized advantages and returns.

    ``segment_ends`` marks boundaries between independently collected
    trajectories that were concatenated into one rollout.  At such a
    boundary, ``segment_next_values`` supplies the bootstrap value for the
    segment that just ended, and GAE propagation is reset so advantages from
    a later segment cannot leak into an earlier one.
    """
    if (segment_ends is None) != (segment_next_values is None):
        raise ValueError(
            "segment_ends and segment_next_values must be supplied together"
        )
    if (
        segment_ends is not None
        and segment_next_values is not None
        and (
            segment_ends.ndim != 1
            or segment_next_values.ndim != 1
            or len(segment_ends) != len(rewards)
            or len(segment_next_values) != len(rewards)
        )
    ):
        raise ValueError("segment metadata must align with rewards")
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        is_segment_end = segment_ends is not None and bool(segment_ends[index])
        if is_segment_end:
            assert segment_next_values is not None
            following_value = float(segment_next_values[index])
            following_gae = 0.0
        else:
            following_value = (
                next_value if index == len(rewards) - 1 else values[index + 1]
            )
            following_gae = last_gae
        non_terminal = 0.0 if dones[index] else 1.0
        delta = rewards[index] + gamma * following_value * non_terminal - values[index]
        last_gae = delta + gamma * gae_lambda * non_terminal * following_gae
        advantages[index] = last_gae
    return advantages, advantages + values


def baseline_factories(*, random_seed: int = 101) -> dict[str, AgentFactory]:
    """Return fresh Phase 4 baseline policies for PPO opponent sampling."""
    return {
        "random": lambda: RandomLegalAgent(seed=random_seed),
        "threshold": lambda: FixedThresholdAgent(threshold=15.0),
        "risk": lambda: BustProbabilityAgent(risk_tolerance=0.20),
        "ev": lambda: ExpectedValueAgent(),
        "dp": lambda: RoundDPAgent(),
    }


class PPOTrainer:
    """Train a masked actor-critic policy against frozen baseline opponents."""

    def __init__(
        self,
        config: PPOConfig,
        *,
        opponent_names: tuple[str, ...] = ("random", "threshold", "risk", "ev", "dp"),
        opponent_provider: OpponentProvider | None = None,
        seat_opponent_provider: SeatOpponentProvider | None = None,
    ) -> None:
        if config.player_count < 3:
            raise ValueError("PPO requires at least three players")
        if not 0 <= config.learner_id < config.player_count:
            raise ValueError("learner_id must reference a seated player")
        if config.learner_seat_mode not in {"fixed", "random"}:
            raise ValueError("learner_seat_mode must be 'fixed' or 'random'")
        if config.network not in {"shared", "separate"}:
            raise ValueError("network must be 'shared' or 'separate'")
        if config.critic_seat_conditioned and config.network != "separate":
            raise ValueError("seat-conditioned critics require the separate network")
        if not opponent_names:
            raise ValueError("at least one opponent is required")
        if config.learning_rate <= 0 or not np.isfinite(config.learning_rate):
            raise ValueError("learning_rate must be positive and finite")
        if config.learning_rate_end is not None and (
            config.learning_rate_end <= 0 or not np.isfinite(config.learning_rate_end)
        ):
            raise ValueError("learning_rate_end must be positive and finite")
        if config.entropy_coefficient < 0 or not np.isfinite(
            config.entropy_coefficient
        ):
            raise ValueError("entropy_coefficient must be non-negative and finite")
        if config.entropy_coefficient_end is not None and (
            config.entropy_coefficient_end < 0
            or not np.isfinite(config.entropy_coefficient_end)
        ):
            raise ValueError("entropy_coefficient_end must be non-negative and finite")
        if config.target_kl is not None and (
            config.target_kl <= 0 or not np.isfinite(config.target_kl)
        ):
            raise ValueError("target_kl must be positive and finite")
        if config.value_clip_epsilon is not None and (
            config.value_clip_epsilon <= 0 or not np.isfinite(config.value_clip_epsilon)
        ):
            raise ValueError("value_clip_epsilon must be positive and finite")
        available = baseline_factories()
        missing = set(opponent_names) - set(available)
        if missing:
            raise ValueError(f"unknown baseline opponents: {sorted(missing)}")
        self.config = config
        self.opponent_names = opponent_names
        self._factories = available
        self._opponent_provider = opponent_provider or self._default_opponent_provider
        self._seat_opponent_provider = seat_opponent_provider
        self._current_update = 1
        seed_torch(config.seed)
        self._rng = random.Random(config.seed)
        initial_env = self.new_env()
        shape = initial_env.observation_space.shape
        if shape is None:
            raise ValueError("environment must expose a fixed observation shape")
        self.observation_size = int(np.prod(shape))
        action_space = cast(Any, initial_env.action_space)
        self.action_size = int(action_space.n)
        self.network: ActorCritic | SeparateActorCritic
        if config.network == "separate":
            self.network = SeparateActorCritic(
                self.observation_size,
                self.action_size,
                config.hidden_size,
                config.player_count if config.critic_seat_conditioned else 0,
            ).to(config.device)
        else:
            self.network = ActorCritic(
                self.observation_size, self.action_size, config.hidden_size
            ).to(config.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)

    def _scheduled_value(
        self, start: float, end: float | None, update: int | None = None
    ) -> float:
        final = start if end is None else end
        current = self._current_update if update is None else update
        progress = (current - 1) / max(1, self.config.updates - 1)
        progress = min(1.0, max(0.0, progress))
        return start + progress * (final - start)

    def _apply_schedules(self) -> tuple[float, float]:
        learning_rate = self._scheduled_value(
            self.config.learning_rate, self.config.learning_rate_end
        )
        entropy_coefficient = self._scheduled_value(
            self.config.entropy_coefficient,
            self.config.entropy_coefficient_end,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate, entropy_coefficient

    def _critic_context(
        self, seat_ids: NDArray[np.int64] | Tensor | None
    ) -> Tensor | None:
        if not self.config.critic_seat_conditioned:
            return None
        if seat_ids is None:
            raise ValueError("seat ids are required by the conditioned critic")
        seats = torch.as_tensor(seat_ids, dtype=torch.int64, device=self.config.device)
        return F.one_hot(seats, num_classes=self.config.player_count).to(torch.float32)

    def _forward(
        self, observations: Tensor, seat_ids: NDArray[np.int64] | Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        context = self._critic_context(seat_ids)
        if isinstance(self.network, SeparateActorCritic):
            return self.network(observations, context)
        return self.network(observations)

    def new_env(
        self, *, requested_learner_id: int | None = None
    ) -> Flip7VsOpponentsEnv:
        if (
            requested_learner_id is not None
            and self._seat_opponent_provider is not None
        ):
            lineup = self._seat_opponent_provider(
                self._rng, self.config.player_count, requested_learner_id
            )
        else:
            lineup = self._opponent_provider(self._rng, self.config.player_count)
        if not 0 <= lineup.learner_id < self.config.player_count:
            raise ValueError("opponent provider returned an invalid learner seat")
        return Flip7VsOpponentsEnv(
            self.config.player_count,
            learner_id=lineup.learner_id,
            observation=ObservationFamily(self.config.observation),
            opponent_observations=lineup.opponent_observations,
            reward=RewardMode(self.config.reward),
            potential_gamma=self.config.gamma,
            opponents=lineup.opponents,
        )

    def _default_opponent_provider(
        self, rng: random.Random, player_count: int
    ) -> EpisodeLineup:
        opponent_name = rng.choice(self.opponent_names)
        policy_factory = self._factories[opponent_name]
        learner_id = self._learner_id_for_episode(rng)
        opponents = {
            agent_name(seat): policy_factory()
            for seat in range(player_count)
            if seat != learner_id
        }
        opponent_observations = {
            agent_name(seat): ObservationFamily.DECK_AWARE
            for seat in range(player_count)
            if seat != learner_id
        }
        return EpisodeLineup(learner_id, opponents, opponent_observations)

    def _learner_id_for_episode(self, rng: random.Random | None = None) -> int:
        source = self._rng if rng is None else rng
        if self.config.learner_seat_mode == "fixed":
            return self.config.learner_id
        return source.randrange(self.config.player_count)

    def collect_rollout(self) -> Rollout:
        observations: list[NDArray[np.float32]] = []
        masks: list[NDArray[np.int8]] = []
        actions: list[int] = []
        old_log_probs: list[float] = []
        rewards: list[float] = []
        dones: list[bool] = []
        values: list[float] = []
        seat_ids: list[int] = []
        env = self.new_env()
        observation, info = env.reset(seed=self._rng.randrange(2**31))
        for _ in range(self.config.rollout_steps):
            mask = info["action_mask"]
            observation_tensor = torch.as_tensor(
                observation, dtype=torch.float32
            ).unsqueeze(0)
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
            with torch.no_grad():
                logits, value = self._forward(
                    observation_tensor,
                    np.asarray([env.learner_id], dtype=np.int64),
                )
                distribution = Categorical(logits=masked_logits(logits, mask_tensor))
                action_tensor: Tensor = distribution.sample()
            action = int(cast(float, action_tensor.item()))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            next_observation, reward, terminated, truncated, next_info = env.step(
                action
            )
            observations.append(observation.copy())
            masks.append(mask.copy())
            actions.append(action)
            old_log_probs.append(
                float(cast(Tensor, distribution.log_prob(action_tensor)).item())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            )
            rewards.append(float(reward))
            dones.append(bool(terminated or truncated))
            values.append(float(value.item()))
            seat_ids.append(env.learner_id)
            observation, info = next_observation, next_info
            if terminated or truncated:
                env.close()
                env = self.new_env()
                observation, info = env.reset(seed=self._rng.randrange(2**31))
        with torch.no_grad():
            next_value = float(
                self._forward(
                    torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0),
                    np.asarray([env.learner_id], dtype=np.int64),
                )[1].item()
            )
        env.close()
        return Rollout(
            np.asarray(observations, dtype=np.float32),
            np.asarray(masks, dtype=np.int8),
            np.asarray(actions, dtype=np.int64),
            np.asarray(old_log_probs, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            np.asarray(values, dtype=np.float32),
            next_value,
            np.asarray(seat_ids, dtype=np.int64),
        )

    def update(self, rollout: Rollout) -> dict[str, float]:
        advantages, returns = compute_gae(
            rollout.rewards,
            rollout.dones,
            rollout.values,
            rollout.next_value,
            self.config.gamma,
            self.config.gae_lambda,
            segment_ends=rollout.segment_ends,
            segment_next_values=rollout.segment_next_values,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        tensors = {
            "observations": torch.as_tensor(rollout.observations, dtype=torch.float32),
            "masks": torch.as_tensor(rollout.masks, dtype=torch.bool),
            "actions": torch.as_tensor(rollout.actions, dtype=torch.int64),
            "old_log_probs": torch.as_tensor(
                rollout.old_log_probs, dtype=torch.float32
            ),
            "old_values": torch.as_tensor(rollout.values, dtype=torch.float32),
            "advantages": torch.as_tensor(advantages, dtype=torch.float32),
            "returns": torch.as_tensor(returns, dtype=torch.float32),
        }
        if rollout.seat_ids is not None:
            tensors["seat_ids"] = torch.as_tensor(rollout.seat_ids, dtype=torch.int64)
        elif self.config.critic_seat_conditioned:
            raise ValueError("rollout is missing seat ids for the conditioned critic")
        size = len(rollout.rewards)
        learning_rate, entropy_coefficient = self._apply_schedules()
        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "updates": 0.0,
        }
        stop_early = False
        for _ in range(self.config.epochs):
            order = torch.randperm(size)
            for start in range(0, size, self.config.minibatch_size):
                indices = order[start : start + self.config.minibatch_size]
                seat_batch = tensors.get("seat_ids")
                logits, values = self._forward(
                    tensors["observations"][indices],
                    None if seat_batch is None else seat_batch[indices],
                )
                distribution = Categorical(
                    logits=masked_logits(logits, tensors["masks"][indices])
                )
                log_probs = cast(
                    Tensor, distribution.log_prob(tensors["actions"][indices])
                )
                ratio = torch.exp(log_probs - tensors["old_log_probs"][indices])
                unclipped = ratio * tensors["advantages"][indices]
                clipped = (
                    torch.clamp(
                        ratio,
                        1 - self.config.clip_epsilon,
                        1 + self.config.clip_epsilon,
                    )
                    * tensors["advantages"][indices]
                )
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                if self.config.value_clip_epsilon is None:
                    value_loss = (values - tensors["returns"][indices]).pow(2).mean()
                else:
                    value_delta = values - tensors["old_values"][indices]
                    clipped_values = tensors["old_values"][indices] + torch.clamp(
                        value_delta,
                        -self.config.value_clip_epsilon,
                        self.config.value_clip_epsilon,
                    )
                    value_loss = torch.maximum(
                        (values - tensors["returns"][indices]).pow(2),
                        (clipped_values - tensors["returns"][indices]).pow(2),
                    ).mean()
                entropy = distribution.entropy().mean()
                approx_kl = (
                    0.5 * (tensors["old_log_probs"][indices] - log_probs).pow(2).mean()
                )
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > self.config.clip_epsilon)
                    .to(torch.float32)
                    .mean()
                )
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - entropy_coefficient * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()  # pyright: ignore[reportUnknownMemberType]
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()  # pyright: ignore[reportUnknownMemberType]
                totals["policy_loss"] += float(policy_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy"] += float(entropy.item())
                totals["approx_kl"] += float(approx_kl.item())
                totals["clip_fraction"] += float(clip_fraction.item())
                totals["updates"] += 1
                if self.config.target_kl is not None and float(approx_kl.item()) > (
                    1.5 * self.config.target_kl
                ):
                    stop_early = True
                    break
            if stop_early:
                break
        count = max(1.0, totals["updates"])
        return {
            key: value / count for key, value in totals.items() if key != "updates"
        } | {
            "updates": totals["updates"],
            "learning_rate": learning_rate,
            "entropy_coefficient": entropy_coefficient,
            "target_kl_stop": float(stop_early),
        }

    def save_checkpoint(
        self,
        path: Path,
        update: int,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        config = asdict(self.config) | {
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "critic_context_size": (
                self.config.player_count if self.config.critic_seat_conditioned else 0
            ),
        }
        payload: dict[str, object] = {
            "model_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "update": update,
            "config": config,
            "opponent_names": self.opponent_names,
            "seed": self.config.seed,
            "package_version": __version__,
        }
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        torch.save(payload, path)

    def load_checkpoint(self, path: Path) -> int:
        """Restore model and optimizer state, returning the saved update."""
        payload: dict[str, Any] = torch.load(
            path, map_location=self.config.device, weights_only=False
        )
        saved_config = payload["config"]
        if (
            saved_config["observation_size"] != self.observation_size
            or saved_config["action_size"] != self.action_size
            or saved_config.get("network", "shared") != self.config.network
            or saved_config.get("critic_context_size", 0)
            != (self.config.player_count if self.config.critic_seat_conditioned else 0)
        ):
            raise ValueError("checkpoint dimensions do not match the trainer")
        self.network.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        return int(payload["update"])

    def train(self, checkpoint: Path | None = None) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for update in range(1, self.config.updates + 1):
            self._current_update = update
            rollout = self.collect_rollout()
            metrics = self.update(rollout)
            metrics["update"] = float(update)
            metrics["mean_reward"] = float(rollout.rewards.mean())
            history.append(metrics)
            if checkpoint is not None:
                self.save_checkpoint(checkpoint, update)
        return history


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    """Write training history as stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"updates": history}, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "EpisodeLineup",
    "OpponentProvider",
    "SeatOpponentProvider",
    "PPOConfig",
    "PPOTrainer",
    "Rollout",
    "baseline_factories",
    "compute_gae",
    "write_history",
]
