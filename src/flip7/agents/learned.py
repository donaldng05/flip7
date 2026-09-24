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


class SeparateActorCritic(nn.Module):
    """Actor-critic with independent actor and value feature extractors.

    The stable follow-up recipe uses this model so value-function fitting cannot
    reshape the actor representation directly.  ``critic_context_size`` is
    reserved for public context such as the learner seat; it is not consumed by
    the actor.
    """

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_size: int = 128,
        critic_context_size: int = 0,
    ) -> None:
        super().__init__()
        if critic_context_size < 0:
            raise ValueError("critic_context_size must be non-negative")
        self.critic_context_size = critic_context_size
        self.actor_body = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.critic_body = nn.Sequential(
            nn.Linear(observation_size + critic_context_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.actor_body:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.critic_body:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)

    def forward(
        self, observations: Tensor, critic_context: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if self.critic_context_size:
            if critic_context is None:
                raise ValueError("critic context is required for this network")
            if (
                critic_context.ndim != 2
                or critic_context.shape[0] != observations.shape[0]
            ):
                raise ValueError("critic context must align with observation batches")
            critic_input = torch.cat((observations, critic_context), dim=-1)
        else:
            if critic_context is not None:
                raise ValueError("critic context is not configured for this network")
            critic_input = observations
        return self.actor(self.actor_body(observations)), self.critic(
            self.critic_body(critic_input)
        ).squeeze(-1)

    def actor_logits(self, observations: Tensor) -> Tensor:
        """Return policy logits without requiring value-function context."""
        return self.actor(self.actor_body(observations))


PolicyNetwork = ActorCritic | SeparateActorCritic


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
        network: PolicyNetwork,
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
            if isinstance(self.network, SeparateActorCritic):
                logits = self.network.actor_logits(observation_tensor)
            else:
                logits, _ = self.network(observation_tensor)
            legal_logits = masked_logits(logits, mask_tensor)
            if self.deterministic:
                return int(torch.argmax(legal_logits, dim=-1).item())
            probabilities = torch.softmax(legal_logits, dim=-1)
            return int(
                torch.multinomial(probabilities, 1, generator=self._generator).item()
            )

    def reseed(self, seed: int) -> None:
        """Reset stochastic inference for a new explicitly seeded episode."""
        self._generator.manual_seed(seed)

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
    ) -> PPOAgent:
        """Build a policy from an already-loaded checkpoint payload."""
        config = payload["config"]
        network_name = config.get("network", "shared")
        if network_name == "separate":
            network: PolicyNetwork = SeparateActorCritic(
                config["observation_size"],
                config["action_size"],
                config["hidden_size"],
                int(config.get("critic_context_size", 0)),
            )
        elif network_name == "shared":
            network = ActorCritic(
                config["observation_size"],
                config["action_size"],
                config["hidden_size"],
            )
        else:
            raise ValueError(f"unsupported checkpoint network: {network_name}")
        network.load_state_dict(payload["model_state"])
        return cls(network, deterministic=deterministic, seed=seed, device=device)


def seed_torch(seed: int) -> None:
    """Seed PyTorch CPU operations used by a training run."""
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    random.seed(seed)
    np.random.seed(seed)


__all__ = [
    "ActorCritic",
    "PPOAgent",
    "PolicyNetwork",
    "SeparateActorCritic",
    "masked_logits",
    "seed_torch",
]
