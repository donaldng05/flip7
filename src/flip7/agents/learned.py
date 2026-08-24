"""PyTorch policies that implement the shared Flip 7 agent protocol."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from flip7.agents.base import ActionMask, Observation


class ActorCritic(nn.Module):
    """Small feed-forward actor-critic network for fixed-size observations."""

    def __init__(
        self, observation_size: int, action_size: int, hidden_size: int = 128
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        features = self.body(observations)
        return self.actor(features), self.critic(features).squeeze(-1)


def masked_logits(logits: Tensor, action_masks: Tensor) -> Tensor:
    """Make invalid actions impossible while preserving valid logits."""
    if logits.shape != action_masks.shape:
        raise ValueError("logits and action masks must have the same shape")
    if not torch.all(action_masks.any(dim=-1)):
        raise ValueError("every action mask row must contain a legal action")
    return logits.masked_fill(action_masks == 0, float("-inf"))


class PPOAgent:
    """Callable learned policy with deterministic or seeded stochastic inference."""

    def __init__(
        self,
        network: ActorCritic,
        *,
        deterministic: bool = False,
        seed: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.network = network.to(device)
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
            logits, _ = self.network(observation_tensor)
            legal_logits = masked_logits(logits, mask_tensor)
            if self.deterministic:
                return int(torch.argmax(legal_logits, dim=-1).item())
            probabilities = torch.softmax(legal_logits, dim=-1)
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
    ) -> PPOAgent:
        payload: dict[str, Any] = torch.load(
            path, map_location=device, weights_only=False
        )
        config = payload["config"]
        network = ActorCritic(
            config["observation_size"],
            config["action_size"],
            config["hidden_size"],
        )
        network.load_state_dict(payload["model_state"])
        return cls(network, deterministic=deterministic, seed=seed, device=device)


def seed_torch(seed: int) -> None:
    """Seed PyTorch CPU operations used by a training run."""
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    random.seed(seed)
    np.random.seed(seed)


__all__ = ["ActorCritic", "PPOAgent", "masked_logits", "seed_torch"]
