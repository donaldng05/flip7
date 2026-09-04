"""Deterministic behavioural diversity measurements for learned policies."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray

from flip7.agents import PPOAgent, SeparateActorCritic
from flip7.agents.learned import masked_logits
from flip7.envs import Flip7AECEnv, ObservationFamily, action_space_size
from flip7.envs.observations import observation_size


@dataclass(frozen=True, slots=True)
class StateBankEntry:
    """One legal, reproducibly generated decision state."""

    observation: NDArray[np.float32]
    action_mask: NDArray[np.int8]
    seat: int
    episode_seed: int
    state_index: int
    expected_action_probabilities: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class StateBank:
    """Fixed state bank reused by every policy in a comparison."""

    entries: tuple[StateBankEntry, ...]
    seed: int
    player_count: int
    observation: ObservationFamily

    @property
    def size(self) -> int:
        return len(self.entries)

    @property
    def observation_size(self) -> int:
        return observation_size(self.player_count, self.observation)

    @property
    def action_size(self) -> int:
        return action_space_size(self.player_count)

    def as_dict(self) -> dict[str, object]:
        return {
            "size": self.size,
            "seed": self.seed,
            "player_count": self.player_count,
            "observation": self.observation.value,
            "observation_size": self.observation_size,
            "action_size": self.action_size,
        }


class SnapshotLike(Protocol):
    """Minimal snapshot interface needed by diversity analysis."""

    @property
    def policy_id(self) -> str: ...

    @property
    def path(self) -> Path: ...

    @property
    def update(self) -> int: ...


def build_state_bank(
    *,
    states: int = 2_048,
    seed: int = 70_000,
    player_count: int = 3,
    observation: ObservationFamily = ObservationFamily.BASIC,
) -> StateBank:
    """Generate exactly ``states`` legal states with a fixed random walk."""
    if states < 1:
        raise ValueError("states must be positive")
    if player_count < 3:
        raise ValueError("player_count must be at least three")
    env = Flip7AECEnv(player_count, observation=observation)
    entries: list[StateBankEntry] = []
    episode_seed = seed
    try:
        while len(entries) < states:
            env.reset(seed=episode_seed)
            for _ in env.agent_iter(max_iter=100_000):
                observation_value, _reward, terminated, truncated, info = env.last()
                if terminated or truncated:
                    env.step(None)
                    continue
                if observation_value is None:
                    raise RuntimeError("state bank received an empty observation")
                mask = np.asarray(info["action_mask"], dtype=np.int8).copy()
                if not np.any(mask):
                    raise RuntimeError("state bank encountered a state without actions")
                legal = np.flatnonzero(mask)
                probabilities = np.zeros(len(mask), dtype=np.float32)
                probabilities[legal] = 1.0 / len(legal)
                seat = int(env.agent_selection.rsplit("_", 1)[1])
                entries.append(
                    StateBankEntry(
                        observation=np.asarray(
                            observation_value, dtype=np.float32
                        ).copy(),
                        action_mask=mask,
                        seat=seat,
                        episode_seed=episode_seed,
                        state_index=len(entries),
                        expected_action_probabilities=probabilities,
                    )
                )
                env.step(int(legal[(len(entries) * 17 + episode_seed) % len(legal)]))
                if len(entries) >= states:
                    break
            episode_seed += 1
    finally:
        env.close()
    return StateBank(tuple(entries), seed, player_count, observation)


def masked_action_probabilities(
    logits: NDArray[np.float32] | torch.Tensor,
    action_mask: NDArray[np.int8] | torch.Tensor,
) -> NDArray[np.float32]:
    """Return a normalized probability vector over legal actions only."""
    logits_tensor = torch.as_tensor(logits, dtype=torch.float32)
    mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool)
    if logits_tensor.ndim != 1 or mask_tensor.ndim != 1:
        raise ValueError("logits and action_mask must be one-dimensional")
    legal_logits = masked_logits(logits_tensor.unsqueeze(0), mask_tensor.unsqueeze(0))[
        0
    ]
    return torch.softmax(legal_logits, dim=-1).detach().cpu().numpy().astype(np.float32)


def jensen_shannon_divergence(
    first: NDArray[np.float32], second: NDArray[np.float32]
) -> float:
    """Compute the symmetric Jensen-Shannon divergence in nats."""
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    if first_values.shape != second_values.shape:
        raise ValueError("probability vectors must have the same shape")
    if np.any(first_values < 0) or np.any(second_values < 0):
        raise ValueError("probability vectors cannot contain negative values")
    first_total = float(first_values.sum())
    second_total = float(second_values.sum())
    if not math.isclose(first_total, 1.0, abs_tol=1e-5) or not math.isclose(
        second_total, 1.0, abs_tol=1e-5
    ):
        raise ValueError("probability vectors must sum to one")
    midpoint = 0.5 * (first_values + second_values)

    def entropy(values: NDArray[np.float64]) -> float:
        positive = values[values > 0]
        return float(-np.sum(positive * np.log(positive)))

    return (
        entropy(midpoint) - 0.5 * entropy(first_values) - 0.5 * entropy(second_values)
    )


def mean_jensen_shannon_divergence(
    first: NDArray[np.float32], second: NDArray[np.float32]
) -> float:
    """Return mean JS divergence for a batch of probability vectors."""
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    if first_values.ndim != 2 or first_values.shape != second_values.shape:
        raise ValueError("probability batches must have the same two-dimensional shape")
    if np.any(first_values < 0) or np.any(second_values < 0):
        raise ValueError("probability vectors cannot contain negative values")
    if not np.allclose(first_values.sum(axis=1), 1.0, atol=1e-5) or not np.allclose(
        second_values.sum(axis=1), 1.0, atol=1e-5
    ):
        raise ValueError("probability vectors must sum to one")

    def entropy(values: NDArray[np.float64]) -> NDArray[np.float64]:
        positive = values > 0
        logs = np.zeros_like(values)
        np.log(values, out=logs, where=positive)
        return -np.sum(np.where(positive, values * logs, 0.0), axis=1)

    midpoint = 0.5 * (first_values + second_values)
    divergence = (
        entropy(midpoint) - 0.5 * entropy(first_values) - 0.5 * entropy(second_values)
    )
    return float(np.mean(divergence))


def action_entropy(probabilities: NDArray[np.float32]) -> float:
    """Return Shannon entropy for one masked action distribution."""
    values = np.asarray(probabilities, dtype=np.float64)
    positive = values[values > 0]
    return float(-np.sum(positive * np.log(positive)))


def action_support(probabilities: NDArray[np.float32], threshold: float = 1e-6) -> int:
    """Count actions with meaningful probability mass."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    return int(np.count_nonzero(np.asarray(probabilities) > threshold))


@dataclass(frozen=True, slots=True)
class PolicyBehavior:
    """State-bank action signature and aggregate behaviour metrics."""

    policy_id: str
    update: int
    probabilities: NDArray[np.float32]
    deterministic_actions: NDArray[np.int64]
    mean_entropy: float
    mean_support: float

    def as_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "update": self.update,
            "states": int(len(self.deterministic_actions)),
            "mean_action_entropy": self.mean_entropy,
            "mean_action_support": self.mean_support,
        }


def policy_behavior(snapshot: SnapshotLike, state_bank: StateBank) -> PolicyBehavior:
    """Evaluate one checkpoint on a fixed state bank."""
    policy = PPOAgent.from_checkpoint(snapshot.path, deterministic=True, device="cpu")  # type: ignore[arg-type]
    network = policy.network
    network.eval()
    with torch.no_grad():
        observations = torch.as_tensor(
            np.stack([entry.observation for entry in state_bank.entries]),
            dtype=torch.float32,
        )
        masks = torch.as_tensor(
            np.stack([entry.action_mask for entry in state_bank.entries]),
            dtype=torch.bool,
        )
        if isinstance(network, SeparateActorCritic):
            logits = network.actor_logits(observations)
        else:
            logits, _value = network(observations)
        probabilities_tensor = torch.softmax(masked_logits(logits, masks), dim=-1)
        probabilities = probabilities_tensor.cpu().numpy().astype(np.float32)
    actions = np.asarray(cast(Any, np.argmax(probabilities, axis=1)), dtype=np.int64)
    entropies = [action_entropy(vector) for vector in probabilities]
    supports = [float(action_support(vector)) for vector in probabilities]
    return PolicyBehavior(
        snapshot.policy_id,
        snapshot.update,
        probabilities,
        actions,
        float(np.mean(entropies)),
        float(np.mean(supports)),
    )


def pairwise_behavior_metrics(
    behaviors: Sequence[PolicyBehavior],
) -> dict[str, object]:
    """Calculate pairwise JS divergence and deterministic disagreement matrices."""
    names = [behavior.policy_id for behavior in behaviors]
    js: dict[str, dict[str, float]] = {name: {} for name in names}
    disagreement: dict[str, dict[str, float]] = {name: {} for name in names}
    for first in behaviors:
        for second in behaviors:
            js[first.policy_id][second.policy_id] = mean_jensen_shannon_divergence(
                first.probabilities, second.probabilities
            )
            first_actions = cast(list[int], first.deterministic_actions.tolist())
            second_actions = cast(list[int], second.deterministic_actions.tolist())
            disagreement[first.policy_id][second.policy_id] = sum(
                first_action != second_action
                for first_action, second_action in zip(
                    first_actions, second_actions, strict=True
                )
            ) / len(first_actions)
    off_diagonal = [
        value
        for first in names
        for second, value in js[first].items()
        if first != second
    ]
    return {
        "policies": names,
        "pairwise_js_divergence": js,
        "pairwise_action_disagreement": disagreement,
        "mean_pairwise_js_divergence": float(np.mean(off_diagonal))
        if off_diagonal
        else 0.0,
        "mean_pairwise_action_disagreement": float(
            np.mean(
                [
                    disagreement[first][second]
                    for first in names
                    for second in names
                    if first != second
                ]
            )
        )
        if len(names) > 1
        else 0.0,
        "policies_metrics": [behavior.as_dict() for behavior in behaviors],
    }


def analyze_snapshots(
    snapshots: Sequence[SnapshotLike],
    state_bank: StateBank,
    response_signatures: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, object]:
    """Return JSON-compatible diversity metrics for every archived snapshot."""
    behaviors = [policy_behavior(snapshot, state_bank) for snapshot in snapshots]
    metrics = pairwise_behavior_metrics(behaviors)
    if response_signatures:
        names = [behavior.policy_id for behavior in behaviors]
        response_matrix: dict[str, dict[str, float]] = {name: {} for name in names}
        off_diagonal: list[float] = []
        for first in names:
            for second in names:
                first_signature = response_signatures.get(first)
                second_signature = response_signatures.get(second)
                if first_signature is None or second_signature is None:
                    continue
                if len(first_signature) != len(second_signature):
                    raise ValueError("response signatures must have equal dimensions")
                distance = float(
                    np.mean(
                        np.abs(
                            np.asarray(first_signature, dtype=np.float64)
                            - np.asarray(second_signature, dtype=np.float64)
                        )
                    )
                )
                response_matrix[first][second] = distance
                if first != second:
                    off_diagonal.append(distance)
        metrics["pairwise_response_distance"] = response_matrix
        metrics["mean_pairwise_response_distance"] = (
            float(np.mean(off_diagonal)) if off_diagonal else 0.0
        )
        metrics["response_signatures"] = {
            name: list(response_signatures[name])
            for name in names
            if name in response_signatures
        }
    return metrics


def non_transitive_cycles(
    matchup_matrix: Mapping[str, Mapping[str, float]], threshold: float = 0.55
) -> list[list[str]]:
    """Find deterministic three-policy win cycles in a matchup matrix."""
    if not 0.5 < threshold <= 1.0:
        raise ValueError("threshold must be greater than 0.5 and at most 1")
    names = sorted(matchup_matrix)
    cycles: list[list[str]] = []
    for first, second, third in itertools.combinations(names, 3):
        orientations = (
            (first, second, third),
            (first, third, second),
        )
        for left, middle, right in orientations:
            if (
                matchup_matrix.get(left, {}).get(middle, 0.0) >= threshold
                and matchup_matrix.get(middle, {}).get(right, 0.0) >= threshold
                and matchup_matrix.get(right, {}).get(left, 0.0) >= threshold
            ):
                cycles.append([left, middle, right, left])
    return cycles


__all__ = [
    "PolicyBehavior",
    "SnapshotLike",
    "StateBank",
    "StateBankEntry",
    "action_entropy",
    "action_support",
    "analyze_snapshots",
    "build_state_bank",
    "jensen_shannon_divergence",
    "mean_jensen_shannon_divergence",
    "masked_action_probabilities",
    "non_transitive_cycles",
    "pairwise_behavior_metrics",
    "policy_behavior",
]
