"""Seat-balanced PPO training for the Phase 7 stability follow-up."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical

from flip7.agents import Agent, PPOAgent
from flip7.agents.learned import masked_logits
from flip7.envs import ObservationFamily, agent_name
from flip7.training.league import baseline_factories
from flip7.training.league_followup import (
    DiversePolicyLeague,
    FollowUpLeagueConfig,
    write_followup_population,
)
from flip7.training.ppo import (
    EpisodeLineup,
    PPOConfig,
    PPOTrainer,
    Rollout,
    RolloutChunkTask,
    collect_rollout_chunk_in_worker,
    write_history,
)

type SeatLineupProvider = Callable[[random.Random, int, int], EpisodeLineup]


def partition_seat_tasks(
    quotas: Sequence[int], num_tasks: int
) -> list[tuple[int, int]]:
    """Partition seat quotas into (seat_id, steps) chunks across workers.

    Guarantees:
    - Sum of steps for each seat equals quotas[seat].
    - Produces at least max(len(quotas), num_tasks) chunks so all workers are utilized.
    """
    if not quotas:
        raise ValueError("quotas must not be empty")
    k = len(quotas)
    sub_counts = [1] * k
    while sum(sub_counts) < num_tasks:
        largest_seat = max(range(k), key=lambda s: quotas[s] / sub_counts[s])
        sub_counts[largest_seat] += 1
    tasks: list[tuple[int, int]] = []
    for s, q in enumerate(quotas):
        base, rem = divmod(q, sub_counts[s])
        for i in range(sub_counts[s]):
            tasks.append((s, base + int(i < rem)))
    return tasks


def training_response_signature(
    checkpoint: Path,
    *,
    observation: ObservationFamily,
    baseline_names: tuple[str, ...],
    games: int,
    seed: int,
    workers: int = 1,
) -> tuple[float, ...]:
    """Measure a checkpoint against training-only baseline matchups.

    The imports are local so the evaluation package can continue importing the
    training package without creating an import cycle.
    """
    if games < 1:
        raise ValueError("response signature games must be positive")
    from flip7.evaluation.phase6 import PHASE6_MATCHUPS, run_rotated_matchups

    matchups = tuple(
        pair for pair in PHASE6_MATCHUPS if set(pair).issubset(baseline_names)
    )
    if not matchups:
        raise ValueError("baseline roster has no supported training matchups")
    policy_instance: PPOAgent | None = None

    def policy() -> PPOAgent:
        nonlocal policy_instance
        if policy_instance is None:
            policy_instance = PPOAgent.from_checkpoint(checkpoint, deterministic=True)
        return policy_instance

    results = run_rotated_matchups(
        policy,
        baseline_factories(),
        games=games,
        seed_bases=tuple(seed + index * 10_000 for index in range(len(matchups))),
        observation=observation,
        matchups=matchups,
        workers=workers,
        checkpoint_path=checkpoint,
    )
    signature: list[float] = []
    for result in results:
        means = result.mean(result.win_share)
        signature.append(float(means["ppo"]))
    return tuple(signature)


def balanced_seat_quotas(steps: int, player_count: int) -> tuple[int, ...]:
    """Return deterministic, maximally equal transition quotas per seat."""
    if steps < 1:
        raise ValueError("rollout steps must be positive")
    if player_count < 1:
        raise ValueError("player count must be positive")
    base, remainder = divmod(steps, player_count)
    return tuple(base + int(seat < remainder) for seat in range(player_count))


class BalancedBaselineProvider:
    """Build baseline-only lineups for an explicitly requested learner seat."""

    def __init__(self, names: tuple[str, ...]) -> None:
        if not names:
            raise ValueError("at least one baseline opponent is required")
        available = baseline_factories()
        missing = set(names) - set(available)
        if missing:
            raise ValueError(f"unknown baseline opponents: {sorted(missing)}")
        self.names = names

    def episode_lineup(
        self,
        rng: random.Random,
        player_count: int,
        learner_id: int | None = None,
    ) -> EpisodeLineup:
        if learner_id is None:
            learner_id = rng.randrange(player_count)
        if not 0 <= learner_id < player_count:
            raise ValueError("learner_id must reference a seated player")
        selected = rng.choice(self.names)
        policy_seed = rng.randrange(2**31)
        factory = baseline_factories(random_seed=policy_seed)[selected]
        opponents: dict[str, Agent] = {
            agent_name(seat): factory()
            for seat in range(player_count)
            if seat != learner_id
        }
        observations = {
            agent_name(seat): ObservationFamily.DECK_AWARE
            for seat in range(player_count)
            if seat != learner_id
        }
        return EpisodeLineup(learner_id, opponents, observations)

    def episode_lineup_for_seat(
        self, rng: random.Random, player_count: int, learner_id: int
    ) -> EpisodeLineup:
        return self.episode_lineup(rng, player_count, learner_id)


class SeatBalancedPPOTrainer(PPOTrainer):
    """PPO trainer whose rollout transitions are balanced across learner seats."""

    def __init__(
        self,
        config: PPOConfig,
        *,
        opponent_provider: Callable[[random.Random, int], EpisodeLineup],
        seat_opponent_provider: SeatLineupProvider,
    ) -> None:
        self.last_rollout_seat_counts: tuple[int, ...] = ()
        self.last_rollout_reset_seeds: tuple[int, ...] = ()
        self.rollout_schedule: list[dict[str, object]] = []
        if config.rollout_steps < config.player_count:
            raise ValueError("rollout steps must cover every learner seat")
        super().__init__(
            config,
            opponent_names=("random",),
            opponent_provider=opponent_provider,
            seat_opponent_provider=seat_opponent_provider,
        )

    def _collect_rollout_parallel(self) -> Rollout:
        quotas = balanced_seat_quotas(
            self.config.rollout_steps, self.config.player_count
        )
        seat_tasks = partition_seat_tasks(quotas, self.config.rollout_workers)
        weights = {k: v.detach().cpu() for k, v in self.network.state_dict().items()}
        tasks = [
            RolloutChunkTask(
                chunk_index=index,
                seed=self._rng.randrange(2**31),
                steps=chunk_steps,
                player_count=self.config.player_count,
                requested_learner_id=seat_id,
                observation_family=self.config.observation,
                reward_mode=self.config.reward,
                gamma=self.config.gamma,
                weights=weights,
                network_type=self.config.network,
                observation_size=self.observation_size,
                action_size=self.action_size,
                hidden_size=self.config.hidden_size,
                critic_seat_conditioned=self.config.critic_seat_conditioned,
                opponent_provider=self._seat_opponent_provider,
            )
            for index, (seat_id, chunk_steps) in enumerate(seat_tasks)
        ]

        if self._worker_pool is not None:
            results = list(
                self._worker_pool.map(collect_rollout_chunk_in_worker, tasks)
            )
        else:
            with ProcessPoolExecutor(max_workers=self.config.rollout_workers) as pool:
                results = list(pool.map(collect_rollout_chunk_in_worker, tasks))

        results.sort(key=lambda r: r.chunk_index)
        total_steps = sum(len(r.rewards) for r in results)
        segment_ends = np.zeros(total_steps, dtype=np.bool_)
        segment_bootstraps = np.zeros(total_steps, dtype=np.float32)
        offset = 0
        all_reset_seeds: list[int] = []
        for r in results:
            all_reset_seeds.extend(r.reset_seeds)
            chunk_len = len(r.rewards)
            end_idx = offset + chunk_len - 1
            segment_ends[end_idx] = True
            segment_bootstraps[end_idx] = r.next_value
            offset += chunk_len

        self.last_rollout_seat_counts = quotas
        self.last_rollout_reset_seeds = tuple(all_reset_seeds)
        self.rollout_schedule.append(
            {
                "rollout": len(self.rollout_schedule) + 1,
                "seat_quotas": list(quotas),
                "reset_seeds": list(all_reset_seeds),
            }
        )

        return Rollout(
            np.concatenate([r.observations for r in results], axis=0),
            np.concatenate([r.masks for r in results], axis=0),
            np.concatenate([r.actions for r in results], axis=0),
            np.concatenate([r.old_log_probs for r in results], axis=0),
            np.concatenate([r.rewards for r in results], axis=0),
            np.concatenate([r.dones for r in results], axis=0),
            np.concatenate([r.values for r in results], axis=0),
            float(results[-1].next_value),
            np.concatenate([r.seat_ids for r in results], axis=0),
            segment_ends,
            segment_bootstraps,
        )

    def collect_rollout(self) -> Rollout:
        """Collect equal-as-possible transition blocks for each learner seat."""
        if self.config.rollout_workers > 1:
            return self._collect_rollout_parallel()
        quotas = balanced_seat_quotas(
            self.config.rollout_steps, self.config.player_count
        )
        observations: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        actions: list[int] = []
        old_log_probs: list[float] = []
        rewards: list[float] = []
        dones: list[bool] = []
        values: list[float] = []
        seat_ids: list[int] = []
        segment_end_indices: list[int] = []
        segment_next_values: list[float] = []
        reset_seeds: list[int] = []
        observation = np.zeros(self.observation_size, dtype=np.float32)

        for learner_id, quota in enumerate(quotas):
            env = self.new_env(requested_learner_id=learner_id)
            try:
                reset_seed = self._rng.randrange(2**31)
                reset_seeds.append(reset_seed)
                observation, info = env.reset(seed=reset_seed)
                for _ in range(quota):
                    mask = info["action_mask"]
                    observation_tensor = torch.as_tensor(
                        observation, dtype=torch.float32
                    ).unsqueeze(0)
                    mask_tensor = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
                    with torch.no_grad():
                        logits, value = self._forward(
                            observation_tensor,
                            np.asarray([learner_id], dtype=np.int64),
                        )
                        distribution = Categorical(
                            logits=masked_logits(logits, mask_tensor)
                        )
                        action_tensor: Tensor = distribution.sample()
                    action = int(cast(float, action_tensor.item()))
                    next_observation, reward, terminated, truncated, next_info = (
                        env.step(action)
                    )
                    observations.append(observation.copy())
                    masks.append(mask.copy())
                    actions.append(action)
                    old_log_probs.append(
                        float(cast(Tensor, distribution.log_prob(action_tensor)).item())
                    )
                    rewards.append(float(reward))
                    dones.append(terminated or truncated)
                    values.append(float(value.item()))
                    seat_ids.append(learner_id)
                    observation, info = next_observation, next_info
                    if terminated or truncated:
                        env.close()
                        env = self.new_env(requested_learner_id=learner_id)
                        reset_seed = self._rng.randrange(2**31)
                        reset_seeds.append(reset_seed)
                        observation, info = env.reset(seed=reset_seed)
                segment_end_indices.append(len(rewards) - 1)
                if dones[-1]:
                    segment_next_values.append(0.0)
                else:
                    with torch.no_grad():
                        segment_next_values.append(
                            float(
                                self._forward(
                                    torch.as_tensor(
                                        observation, dtype=torch.float32
                                    ).unsqueeze(0),
                                    np.asarray([learner_id], dtype=np.int64),
                                )[1].item()
                            )
                        )
            finally:
                env.close()

        if not observations:
            raise RuntimeError("balanced rollout collected no transitions")
        segment_ends = np.zeros(len(rewards), dtype=np.bool_)
        segment_bootstraps = np.zeros(len(rewards), dtype=np.float32)
        for index, bootstrap in zip(
            segment_end_indices, segment_next_values, strict=True
        ):
            segment_ends[index] = True
            segment_bootstraps[index] = bootstrap
        next_value = float(segment_next_values[-1])
        self.last_rollout_seat_counts = quotas
        self.last_rollout_reset_seeds = tuple(reset_seeds)
        self.rollout_schedule.append(
            {
                "rollout": len(self.rollout_schedule) + 1,
                "seat_quotas": list(quotas),
                "reset_seeds": list(reset_seeds),
            }
        )
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
            segment_ends,
            segment_bootstraps,
        )


class StabilityLeaguePPOTrainer(SeatBalancedPPOTrainer):
    """Train a league policy with balanced seats and a gradual opponent curriculum."""

    def __init__(
        self,
        config: PPOConfig,
        *,
        league_config: FollowUpLeagueConfig,
        workers: int = 1,
    ) -> None:
        self.workers = workers
        self.league = DiversePolicyLeague(
            league_config,
            observation=ObservationFamily(config.observation),
            source_seed=config.seed,
            player_count=config.player_count,
        )
        super().__init__(
            config,
            opponent_provider=self.league.episode_lineup,
            seat_opponent_provider=self.league.episode_lineup_for_seat,
        )

    def train(self, checkpoint: Path | None = None) -> list[dict[str, float]]:
        """Train and archive snapshots using the fixed final update contract."""
        if checkpoint is None:
            raise ValueError("StabilityLeaguePPOTrainer requires a checkpoint path")
        with self.worker_pool_scope():
            history: list[dict[str, float]] = []
            snapshot_dir = checkpoint.parent / "checkpoints"
            for update in range(1, self.config.updates + 1):
                self._current_update = update
                self.league.set_training_update(update)
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
                    response_signature = None
                    if self.league.config.response_signature_games > 0:
                        response_signature = training_response_signature(
                            snapshot_path,
                            observation=ObservationFamily(self.config.observation),
                            baseline_names=self.league.config.baseline_names,
                            games=self.league.config.response_signature_games,
                            seed=self.config.seed * 100_000 + update,
                            workers=self.workers,
                        )
                    self.league.register_snapshot(
                        snapshot_path,
                        update=update,
                        seed=self.config.seed,
                        response_signature=response_signature,
                    )
                metrics.update(
                    {
                        "population_size": float(len(self.league.snapshots)),
                        "archived_population_size": float(
                            len(self.league.archived_snapshots)
                        ),
                        "learned_opponent_probability": self.league.learned_probability,
                        "seat_0_transitions": float(self.last_rollout_seat_counts[0]),
                        "seat_1_transitions": float(self.last_rollout_seat_counts[1]),
                        "seat_2_transitions": float(self.last_rollout_seat_counts[2]),
                    }
                )
                total = sum(self.league.exposure.values())
                learned = sum(
                    count
                    for key, count in self.league.exposure.items()
                    if key.startswith("snapshot:")
                )
                metrics["learned_opponent_share"] = learned / total if total else 0.0
                history.append(metrics)
            return history


class StabilityControlPPOTrainer(SeatBalancedPPOTrainer):
    """Train a balanced baseline-only control with the same PPO recipe."""

    def __init__(self, config: PPOConfig, baseline_names: tuple[str, ...]) -> None:
        provider = BalancedBaselineProvider(baseline_names)
        super().__init__(
            config,
            opponent_provider=lambda rng, count: provider.episode_lineup(rng, count),
            seat_opponent_provider=provider.episode_lineup_for_seat,
        )

    def train(self, checkpoint: Path | None = None) -> list[dict[str, float]]:
        if checkpoint is None:
            raise ValueError("StabilityControlPPOTrainer requires a checkpoint path")
        with self.worker_pool_scope():
            history: list[dict[str, float]] = []
            for update in range(1, self.config.updates + 1):
                self._current_update = update
                rollout = self.collect_rollout()
                metrics = self.update(rollout)
                metrics["update"] = float(update)
                metrics["mean_reward"] = float(rollout.rewards.mean())
                self.save_checkpoint(checkpoint, update)
                metrics.update(
                    {
                        "seat_0_transitions": float(self.last_rollout_seat_counts[0]),
                        "seat_1_transitions": float(self.last_rollout_seat_counts[1]),
                        "seat_2_transitions": float(self.last_rollout_seat_counts[2]),
                    }
                )
                history.append(metrics)
            return history


__all__ = [
    "BalancedBaselineProvider",
    "SeatBalancedPPOTrainer",
    "StabilityControlPPOTrainer",
    "StabilityLeaguePPOTrainer",
    "balanced_seat_quotas",
    "partition_seat_tasks",
    "training_response_signature",
    "write_followup_population",
    "write_history",
]
