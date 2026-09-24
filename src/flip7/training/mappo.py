"""A deliberately small, masked MAPPO pilot for the native AEC environment."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn, optim
from torch.distributions import Categorical

from flip7 import __version__
from flip7.agents import ActionMask, Observation, seed_torch
from flip7.agents.learned import masked_logits
from flip7.envs import (
    Flip7AECEnv,
    ObservationFamily,
    RewardMode,
    action_space_size,
)
from flip7.envs.observations import encode_observation, observation_size


@dataclass(frozen=True, slots=True)
class MAPPOConfig:
    """Matched hyperparameters for the focused centralized-critic pilot."""

    seed: int = 7
    player_count: int = 3
    observation: str = "basic"
    reward: str = "sparse_win"
    hidden_size: int = 128
    rollout_steps: int = 1_024
    updates: int = 100
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

    def __post_init__(self) -> None:
        if self.player_count < 3:
            raise ValueError("MAPPO requires at least three players")
        if ObservationFamily(self.observation) is not ObservationFamily.BASIC:
            raise ValueError("the MAPPO pilot uses the basic observation")
        if self.rollout_steps < 1 or self.updates < 1:
            raise ValueError("rollout_steps and updates must be positive")
        if self.epochs < 1 or self.minibatch_size < 1:
            raise ValueError("epochs and minibatch_size must be positive")
        RewardMode(self.reward)


class SharedActor(nn.Module):
    """Feed-forward actor shared by all trainable seats."""

    def __init__(self, observation_size_value: int, action_size: int, hidden_size: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_size_value, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_size)

    def forward(self, observations: Tensor) -> Tensor:
        return self.actor(self.body(observations))


class CentralizedCritic(nn.Module):
    """Value function over all seats and AEC active-seat metadata."""

    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, observations: Tensor) -> Tensor:
        return self.body(observations).squeeze(-1)


@dataclass(frozen=True, slots=True)
class MAPPORollout:
    observations: NDArray[np.float32]
    masks: NDArray[np.int8]
    global_observations: NDArray[np.float32]
    actions: NDArray[np.int64]
    old_log_probs: NDArray[np.float32]
    rewards: NDArray[np.float32]
    dones: NDArray[np.bool_]
    values: NDArray[np.float32]
    next_values: NDArray[np.float32]


class MAPPOAgent:
    """Callable shared actor used for evaluation of a MAPPO checkpoint."""

    def __init__(
        self,
        actor: SharedActor,
        *,
        deterministic: bool = True,
        seed: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.actor = actor.to(device)
        self.actor.eval()
        self.deterministic = deterministic
        self.device = torch.device(device)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(0 if seed is None else seed)

    def __call__(self, observation: Observation, action_mask: ActionMask) -> int:
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        mask_tensor = torch.as_tensor(
            action_mask, dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits = masked_logits(self.actor(observation_tensor), mask_tensor)
            if self.deterministic:
                return int(torch.argmax(logits, dim=-1).item())
            probabilities = torch.softmax(logits, dim=-1)
            return int(
                torch.multinomial(probabilities, 1, generator=self._generator).item()
            )

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        *,
        deterministic: bool = True,
        seed: int | None = None,
        device: str = "cpu",
    ) -> MAPPOAgent:
        payload: dict[str, Any] = torch.load(
            path, map_location=device, weights_only=False
        )
        return cls.from_payload(
            payload, deterministic=deterministic, seed=seed, device=device
        )

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        deterministic: bool = True,
        seed: int | None = None,
        device: str = "cpu",
    ) -> MAPPOAgent:
        """Build a policy from an already-loaded checkpoint payload."""
        config = cast(dict[str, Any], payload["config"])
        actor = SharedActor(
            config["observation_size"], config["action_size"], config["hidden_size"]
        )
        actor.load_state_dict(payload["actor_state"])
        return cls(actor, deterministic=deterministic, seed=seed, device=device)


class MAPPOTrainer:
    """Train three shared-policy seats with a centralized value critic."""

    def __init__(self, config: MAPPOConfig):
        self.config = config
        seed_torch(config.seed)
        self._rng = random.Random(config.seed)
        self.observation_size = observation_size(
            config.player_count, ObservationFamily.BASIC
        )
        self.action_size = action_space_size(config.player_count)
        self.critic_input_size = (
            config.player_count * self.observation_size + config.player_count + 1
        )
        self.actor = SharedActor(
            self.observation_size, self.action_size, config.hidden_size
        ).to(config.device)
        self.critic = CentralizedCritic(self.critic_input_size, config.hidden_size).to(
            config.device
        )
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.learning_rate,
        )

    def _critic_observation(self, env: Flip7AECEnv) -> NDArray[np.float32]:
        observations = np.concatenate(
            [
                encode_observation(env.engine.state, seat, ObservationFamily.BASIC)
                for seat in range(self.config.player_count)
            ]
        )
        active = np.zeros(self.config.player_count, dtype=np.float32)
        active[int(env.agent_selection.rsplit("_", 1)[1])] = 1.0
        metadata = np.asarray([env.engine.state.round_number / 30.0], dtype=np.float32)
        return np.concatenate((observations, active, metadata)).astype(np.float32)

    def collect_rollout(self) -> MAPPORollout:
        observations: list[NDArray[np.float32]] = []
        masks: list[NDArray[np.int8]] = []
        global_observations: list[NDArray[np.float32]] = []
        actions: list[int] = []
        old_log_probs: list[float] = []
        rewards: list[float] = []
        dones: list[bool] = []
        values: list[float] = []
        next_values: list[float] = []
        env = Flip7AECEnv(
            self.config.player_count,
            observation=ObservationFamily.BASIC,
            reward=RewardMode(self.config.reward),
        )
        try:
            while len(actions) < self.config.rollout_steps:
                env.reset(seed=self._rng.randrange(2**31))
                for _ in env.agent_iter(max_iter=100_000):
                    observation_value, _reward, terminated, truncated, info = env.last()
                    if terminated or truncated:
                        env.step(None)
                        continue
                    if observation_value is None:
                        raise RuntimeError("MAPPO received an empty active observation")
                    observation = np.asarray(observation_value, dtype=np.float32).copy()
                    mask = np.asarray(info["action_mask"], dtype=np.int8).copy()
                    global_observation = self._critic_observation(env)
                    with torch.no_grad():
                        obs_tensor = torch.as_tensor(
                            observation, dtype=torch.float32
                        ).unsqueeze(0)
                        mask_tensor = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(
                            0
                        )
                        distribution = Categorical(
                            logits=masked_logits(self.actor(obs_tensor), mask_tensor)
                        )
                        action_tensor = distribution.sample()
                        value = self.critic(
                            torch.as_tensor(
                                global_observation, dtype=torch.float32
                            ).unsqueeze(0)
                        )
                    action = int(action_tensor.item())
                    acting_agent = env.agent_selection
                    env.step(action)
                    done = bool(env.engine.state.is_game_terminal)
                    # ``env.rewards`` is assigned immediately after the transition;
                    # read the reward belonging to the seat that just acted.
                    reward = float(env.rewards.get(acting_agent, 0.0))
                    if done:
                        next_global = np.zeros(self.critic_input_size, dtype=np.float32)
                        next_value = 0.0
                    else:
                        next_global = self._critic_observation(env)
                        with torch.no_grad():
                            next_value = float(
                                self.critic(
                                    torch.as_tensor(
                                        next_global, dtype=torch.float32
                                    ).unsqueeze(0)
                                ).item()
                            )
                    observations.append(observation)
                    masks.append(mask)
                    global_observations.append(global_observation)
                    actions.append(action)
                    old_log_probs.append(
                        float(cast(Tensor, distribution.log_prob(action_tensor)).item())
                    )
                    rewards.append(reward)
                    dones.append(done)
                    values.append(float(value.item()))
                    next_values.append(next_value)
                    if done or len(actions) >= self.config.rollout_steps:
                        break
                if len(actions) >= self.config.rollout_steps:
                    break
        finally:
            env.close()
        return MAPPORollout(
            np.asarray(observations, dtype=np.float32),
            np.asarray(masks, dtype=np.int8),
            np.asarray(global_observations, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(old_log_probs, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            np.asarray(values, dtype=np.float32),
            np.asarray(next_values, dtype=np.float32),
        )

    def update(self, rollout: MAPPORollout) -> dict[str, float]:
        advantages = np.zeros_like(rollout.rewards, dtype=np.float32)
        last = 0.0
        for index in range(len(advantages) - 1, -1, -1):
            non_terminal = 0.0 if rollout.dones[index] else 1.0
            delta = (
                rollout.rewards[index]
                + self.config.gamma * rollout.next_values[index] * non_terminal
                - rollout.values[index]
            )
            last = (
                delta + self.config.gamma * self.config.gae_lambda * non_terminal * last
            )
            advantages[index] = last
        returns = advantages + rollout.values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        tensors = {
            "observations": torch.as_tensor(rollout.observations, dtype=torch.float32),
            "masks": torch.as_tensor(rollout.masks, dtype=torch.bool),
            "global": torch.as_tensor(rollout.global_observations, dtype=torch.float32),
            "actions": torch.as_tensor(rollout.actions, dtype=torch.int64),
            "old_log_probs": torch.as_tensor(
                rollout.old_log_probs, dtype=torch.float32
            ),
            "advantages": torch.as_tensor(advantages, dtype=torch.float32),
            "returns": torch.as_tensor(returns, dtype=torch.float32),
        }
        size = len(rollout.rewards)
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0.0}
        for _ in range(self.config.epochs):
            order = torch.randperm(size)
            for start in range(0, size, self.config.minibatch_size):
                indices = order[start : start + self.config.minibatch_size]
                distribution = Categorical(
                    logits=masked_logits(
                        self.actor(tensors["observations"][indices]),
                        tensors["masks"][indices],
                    )
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
                values = self.critic(tensors["global"][indices])
                value_loss = (values - tensors["returns"][indices]).pow(2).mean()
                entropy = distribution.entropy().mean()
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()  # pyright: ignore[reportUnknownMemberType]
                totals["policy_loss"] += float(policy_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy"] += float(entropy.item())
                totals["updates"] += 1
        count = max(1.0, totals["updates"])
        return {
            key: value / count for key, value in totals.items() if key != "updates"
        } | {"updates": totals["updates"]}

    def save_checkpoint(self, path: Path, update: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor_state": self.actor.state_dict(),
                "critic_state": self.critic.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "update": update,
                "config": asdict(self.config)
                | {
                    "observation_size": self.observation_size,
                    "action_size": self.action_size,
                    "critic_input_size": self.critic_input_size,
                },
                "seed": self.config.seed,
                "package_version": __version__,
            },
            path,
        )

    def train(self, checkpoint: Path | None = None) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for update in range(1, self.config.updates + 1):
            rollout = self.collect_rollout()
            metrics = self.update(rollout)
            metrics["update"] = float(update)
            metrics["mean_reward"] = float(rollout.rewards.mean())
            history.append(metrics)
            if checkpoint is not None:
                self.save_checkpoint(checkpoint, update)
        return history


def write_mappo_history(path: Path, history: list[dict[str, float]]) -> None:
    """Write MAPPO training history using the same JSON convention as PPO."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"updates": history}, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "CentralizedCritic",
    "MAPPOAgent",
    "MAPPOConfig",
    "MAPPORollout",
    "MAPPOTrainer",
    "SharedActor",
    "write_mappo_history",
]
