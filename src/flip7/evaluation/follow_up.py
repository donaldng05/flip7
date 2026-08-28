"""Paired schedules and comparisons for the Phase 7 follow-up."""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from flip7.agents import AgentFactory, PPOAgent
from flip7.envs import ObservationFamily, RewardMode
from flip7.evaluation.metrics import GameResult, MatchupMetrics
from flip7.evaluation.phase7 import (
    EloTable,
    TournamentParticipant,
    TournamentResult,
)
from flip7.evaluation.tournament import run_game


@dataclass(frozen=True, slots=True)
class ScheduledGame:
    """One canonical lineup, seat permutation, and deterministic game seed."""

    lineup: tuple[str, str, str]
    seats: tuple[str, str, str]
    game_index: int
    seed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "lineup": list(self.lineup),
            "seats": list(self.seats),
            "game_index": self.game_index,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class _ParallelParticipant:
    """Serializable participant description used by tournament workers."""

    name: str
    observation: ObservationFamily
    checkpoint: str | None


_worker_factories: dict[str, AgentFactory] = {}
_worker_observations: dict[str, ObservationFamily] = {}
_worker_reward = RewardMode.SPARSE_WIN


def _initialize_tournament_worker(
    participants: tuple[_ParallelParticipant, ...], reward: str
) -> None:
    """Load one immutable copy of every tournament policy in a worker."""
    global _worker_factories, _worker_observations, _worker_reward

    # Avoid multiplying Torch's intra-op thread pool by the process count.
    import torch

    torch.set_num_threads(1)
    from flip7.training.ppo import baseline_factories

    factories = baseline_factories()
    observations: dict[str, ObservationFamily] = {}
    for participant in participants:
        observations[participant.name] = participant.observation
        if participant.checkpoint is None:
            if participant.name not in factories:
                raise ValueError(
                    f"parallel tournament participant has no checkpoint: "
                    f"{participant.name}"
                )
            continue
        policy = PPOAgent.from_checkpoint(
            Path(participant.checkpoint), deterministic=True, device="cpu"
        )
        factories[participant.name] = lambda policy=policy: policy
    _worker_factories = factories
    _worker_observations = observations
    _worker_reward = RewardMode(reward)


def _run_scheduled_game_in_worker(item: ScheduledGame) -> GameResult:
    """Execute one scheduled game using policies cached in the worker."""
    ordered = tuple(_worker_factories[name] for name in item.seats)
    result = run_game(
        ordered,
        seed=item.seed,
        observation=ObservationFamily.DECK_AWARE,
        reward=_worker_reward,
        agent_observations=tuple(_worker_observations[name] for name in item.seats),
    )
    return GameResult(
        agents=item.seats,
        player_count=result.player_count,
        winning_player_ids=result.winning_player_ids,
        final_scores=result.final_scores,
        round_scores=result.round_scores,
        rounds=result.rounds,
        busted_rounds=result.busted_rounds,
        flip7_events=result.flip7_events,
        action_counts=result.action_counts,
    )


def _parallel_participants(
    participants: Sequence[TournamentParticipant],
    checkpoint_paths: Mapping[str, Path],
) -> tuple[_ParallelParticipant, ...]:
    """Convert named participants to the process-safe tournament contract."""
    return tuple(
        _ParallelParticipant(
            participant.name,
            participant.observation,
            str(checkpoint_paths[participant.name])
            if participant.name in checkpoint_paths
            else None,
        )
        for participant in participants
    )


def _stable_seed(base: int, lineup: tuple[str, str, str], game_index: int) -> int:
    digest = hashlib.sha256("|".join(lineup).encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big")
    return base + offset + game_index


def build_paired_schedule(
    names: Sequence[str], *, games: int = 10, seed: int = 20_000
) -> tuple[ScheduledGame, ...]:
    """Build a participant-order-independent schedule with all seat rotations."""
    if len(names) < 3 or len(set(names)) != len(names):
        raise ValueError("a schedule requires at least three unique participants")
    if games < 1:
        raise ValueError("games must be positive")
    canonical_names = tuple(sorted(names))
    scheduled: list[ScheduledGame] = []
    for lineup in itertools.combinations(canonical_names, 3):
        typed_lineup = (lineup[0], lineup[1], lineup[2])
        for game_index in range(games):
            game_seed = _stable_seed(seed, typed_lineup, game_index)
            for seats in itertools.permutations(lineup):
                typed_seats = (seats[0], seats[1], seats[2])
                scheduled.append(
                    ScheduledGame(
                        typed_lineup,
                        typed_seats,
                        game_index,
                        game_seed,
                    )
                )
    return tuple(scheduled)


def run_paired_round_robin(
    participants: Sequence[TournamentParticipant],
    *,
    games: int = 10,
    seed: int = 20_000,
    reward: RewardMode = RewardMode.SPARSE_WIN,
    initial_rating: float = 1500.0,
    k_factor: float = 32.0,
    schedule: Sequence[ScheduledGame] | None = None,
    workers: int = 1,
    checkpoint_paths: Mapping[str, Path] | None = None,
) -> tuple[TournamentResult, tuple[ScheduledGame, ...]]:
    """Run a fixed paired tournament and return its reusable schedule.

    ``workers`` enables deterministic process-level parallelism for expensive
    evaluation. Results are collected in schedule order before Elo updates,
    so changing the worker count cannot change the reported tournament.
    ``checkpoint_paths`` is required for learned participants when workers are
    greater than one; baseline factories are reconstructed in each worker.
    """
    if len(participants) < 3:
        raise ValueError("a tournament requires at least three participants")
    by_name = {participant.name: participant for participant in participants}
    if len(by_name) != len(participants):
        raise ValueError("tournament participant names must be unique")
    if workers < 1:
        raise ValueError("workers must be positive")
    selected_schedule = tuple(
        schedule or build_paired_schedule(tuple(by_name), games=games, seed=seed)
    )
    expected_names = set(by_name)
    if any(set(item.lineup) - expected_names for item in selected_schedule):
        raise ValueError("schedule contains an unknown participant")
    elo = EloTable(
        tuple(sorted(by_name)), initial_rating=initial_rating, k_factor=k_factor
    )
    metrics_by_lineup: dict[tuple[str, str, str], MatchupMetrics] = {}
    if workers == 1:
        results = (
            _run_game_for_participants(item, by_name, reward=reward)
            for item in selected_schedule
        )
    else:
        if checkpoint_paths is None:
            raise ValueError(
                "checkpoint_paths are required when tournament workers exceed one"
            )
        worker_participants = _parallel_participants(participants, checkpoint_paths)
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_tournament_worker,
            initargs=(worker_participants, reward.value),
        ) as executor:
            results = executor.map(
                _run_scheduled_game_in_worker,
                selected_schedule,
                chunksize=4,
            )
    for item, named in zip(selected_schedule, results, strict=True):
        lineup_metrics = metrics_by_lineup.setdefault(
            item.lineup, MatchupMetrics(item.lineup, 3)
        )
        lineup_metrics.add(named)
        elo.update(named.agents, named.winning_player_ids)
    result = TournamentResult(
        games=len(selected_schedule),
        lineup_games=games * 6 * len(tuple(itertools.combinations(by_name, 3))),
        matchups=tuple(metrics_by_lineup.values()),
        elo=elo.as_dict(),
    )
    return result, selected_schedule


def _run_game_for_participants(
    item: ScheduledGame,
    participants: Mapping[str, TournamentParticipant],
    *,
    reward: RewardMode,
) -> GameResult:
    """Run one game through the original in-process participant factories."""
    ordered = tuple(participants[name] for name in item.seats)
    result = run_game(
        tuple(participant.factory for participant in ordered),
        seed=item.seed,
        observation=ObservationFamily.DECK_AWARE,
        reward=reward,
        agent_observations=tuple(participant.observation for participant in ordered),
    )
    return GameResult(
        agents=item.seats,
        player_count=result.player_count,
        winning_player_ids=result.winning_player_ids,
        final_scores=result.final_scores,
        round_scores=result.round_scores,
        rounds=result.rounds,
        busted_rounds=result.busted_rounds,
        flip7_events=result.flip7_events,
        action_counts=result.action_counts,
    )


def direct_final_warmup_comparison(
    final: TournamentParticipant,
    warmup: TournamentParticipant,
    opponents: Sequence[TournamentParticipant],
    *,
    games: int = 10,
    seed: int = 40_000,
    reward: RewardMode = RewardMode.SPARSE_WIN,
) -> dict[str, object]:
    """Compare final and warmup policies in paired three-player games."""
    if games < 1:
        raise ValueError("games must be positive")
    rows: list[dict[str, object]] = []
    for opponent in opponents:
        participants = (final, warmup, opponent)
        schedule = build_paired_schedule(
            tuple(item.name for item in participants), games=games, seed=seed
        )
        final_wins = 0.0
        warmup_wins = 0.0
        for item in schedule:
            by_name = {participant.name: participant for participant in participants}
            ordered = tuple(by_name[name] for name in item.seats)
            result = run_game(
                tuple(participant.factory for participant in ordered),
                seed=item.seed,
                observation=ObservationFamily.DECK_AWARE,
                reward=reward,
                agent_observations=tuple(
                    participant.observation for participant in ordered
                ),
            )
            winner_count = len(result.winning_player_ids)
            final_seat = item.seats.index(final.name)
            warmup_seat = item.seats.index(warmup.name)
            share = 1.0 / winner_count if winner_count else 0.0
            final_wins += share if final_seat in result.winning_player_ids else 0.0
            warmup_wins += share if warmup_seat in result.winning_player_ids else 0.0
        total_games = len(schedule)
        rows.append(
            {
                "opponent": opponent.name,
                "games": total_games,
                "final_win_share": final_wins / total_games,
                "warmup_win_share": warmup_wins / total_games,
                "delta": (final_wins - warmup_wins) / total_games,
            }
        )
    pooled_games = sum(cast(int, row["games"]) for row in rows)
    return {
        "games": pooled_games,
        "opponents": rows,
        "final_win_share": sum(
            cast(float, row["final_win_share"]) * cast(int, row["games"])
            for row in rows
        )
        / pooled_games,
        "warmup_win_share": sum(
            cast(float, row["warmup_win_share"]) * cast(int, row["games"])
            for row in rows
        )
        / pooled_games,
        "delta": sum(
            cast(float, row["delta"]) * cast(int, row["games"]) for row in rows
        )
        / pooled_games,
    }


def schedule_as_dict(schedule: Sequence[ScheduledGame]) -> list[dict[str, object]]:
    """Serialize a schedule without relying on Python hash randomization."""
    return [item.as_dict() for item in schedule]


def validate_followup_manifest(manifest: Mapping[str, object]) -> None:
    """Validate the complete artifact contract for a follow-up run."""
    required = (
        "checkpoint",
        "training_history",
        "population",
        "diversity",
        "evaluation",
        "tournament",
        "manifest",
    )
    for key in required:
        value = manifest.get(key)
        if not isinstance(value, str) or not Path(value).is_file():
            raise ValueError(f"follow-up manifest is missing a file for {key}")


def matchup_win_share_matrix(result: TournamentResult) -> dict[str, dict[str, float]]:
    """Project three-player lineup results into a comparable pair matrix.

    Each cell is the mean normalized win share of the row policy against the
    column policy across lineups in which both appeared.  It is a diagnostic
    matrix, not a replacement for the full three-player tournament result.
    """
    totals: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for matchup in result.matchups:
        means = matchup.mean(matchup.win_share)
        for first, second in itertools.combinations(matchup.agents, 2):
            denominator = means[first] + means[second]
            first_share = means[first] / denominator if denominator else 0.5
            second_share = 1.0 - first_share
            totals.setdefault(first, {}).setdefault(second, 0.0)
            totals.setdefault(second, {}).setdefault(first, 0.0)
            counts.setdefault(first, {}).setdefault(second, 0)
            counts.setdefault(second, {}).setdefault(first, 0)
            totals[first][second] += first_share
            totals[second][first] += second_share
            counts[first][second] += 1
            counts[second][first] += 1
    return {
        first: {
            second: totals[first][second] / counts[first][second]
            for second in sorted(totals[first])
        }
        for first in sorted(totals)
    }


__all__ = [
    "ScheduledGame",
    "build_paired_schedule",
    "direct_final_warmup_comparison",
    "matchup_win_share_matrix",
    "run_paired_round_robin",
    "schedule_as_dict",
    "validate_followup_manifest",
]
