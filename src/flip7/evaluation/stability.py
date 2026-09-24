"""Paired evaluation primitives for the Phase 7 stability follow-up."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from flip7.agents import AgentFactory
from flip7.envs import ObservationFamily
from flip7.evaluation.metrics import GameResult
from flip7.evaluation.phase6 import PHASE6_MATCHUPS
from flip7.evaluation.tournament import load_worker_policy, run_game


@dataclass(frozen=True, slots=True)
class ScheduledPairedGame:
    """One scheduled paired comparison to be executed in a worker process."""

    matchup_index: int
    learner_seat: int
    game_index: int
    seed: int
    matchup: str
    first_lineup: tuple[str, ...]
    second_lineup: tuple[str, ...]
    first_observations: tuple[ObservationFamily, ...]
    second_observations: tuple[ObservationFamily, ...]
    first_family: ObservationFamily
    second_family: ObservationFamily


_worker_roster: dict[str, AgentFactory] = {}


def _initialize_paired_worker(
    checkpoint_paths: tuple[tuple[str, str], ...],
) -> None:
    """Initialize worker process with thread count and cached candidate policies."""
    global _worker_roster
    import torch

    torch.set_num_threads(1)
    from flip7.evaluation.phase7 import heldout_factories
    from flip7.training.ppo import baseline_factories

    factories: dict[str, AgentFactory] = baseline_factories() | heldout_factories()
    for name, path_str in checkpoint_paths:
        policy = load_worker_policy(Path(path_str))
        factories[name] = lambda policy=policy: policy
    _worker_roster = factories


def _run_paired_game_in_worker(item: ScheduledPairedGame) -> PairedGame:
    """Execute one paired game on the same seed in a worker process."""
    first_factories = tuple(_worker_roster[name] for name in item.first_lineup)
    first_result = run_game(
        first_factories,
        seed=item.seed,
        observation=item.first_family,
        agent_observations=item.first_observations,
    )
    second_factories = tuple(_worker_roster[name] for name in item.second_lineup)
    second_result = run_game(
        second_factories,
        seed=item.seed,
        observation=item.second_family,
        agent_observations=item.second_observations,
    )
    return PairedGame(
        seed=item.seed,
        block=item.matchup_index,
        matchup=item.matchup,
        learner_seat=item.learner_seat,
        first_win_share=_learner_share(first_result, item.learner_seat),
        second_win_share=_learner_share(second_result, item.learner_seat),
    )


@dataclass(frozen=True, slots=True)
class PairedGame:
    """One common-seed comparison between two learner checkpoints."""

    seed: int
    block: int
    matchup: str
    learner_seat: int
    first_win_share: float
    second_win_share: float

    @property
    def difference(self) -> float:
        return self.first_win_share - self.second_win_share


@dataclass(frozen=True, slots=True)
class PairedEvaluation:
    """Paired learner comparison with game- and block-level intervals."""

    games: tuple[PairedGame, ...]
    first_name: str = "first"
    second_name: str = "second"

    def as_dict(self) -> dict[str, object]:
        differences = [game.difference for game in self.games]
        by_seed: dict[int, list[float]] = {}
        for game in self.games:
            by_seed.setdefault(game.block, []).append(game.difference)
        seed_means = [sum(values) / len(values) for values in by_seed.values()]
        return {
            "first": self.first_name,
            "second": self.second_name,
            "games": len(self.games),
            "mean_difference": _mean(differences),
            "game_level_95_ci": _normal_interval(differences),
            "block_level_means": seed_means,
            "block_level_95_ci": _normal_interval(seed_means),
            "per_game": [
                {
                    "seed": game.seed,
                    "block": game.block,
                    "matchup": game.matchup,
                    "learner_seat": game.learner_seat,
                    "first_win_share": game.first_win_share,
                    "second_win_share": game.second_win_share,
                    "difference": game.difference,
                }
                for game in self.games
            ],
        }


def run_paired_rotated_evaluation(
    first_factory: AgentFactory,
    second_factory: AgentFactory,
    baseline_roster: Mapping[str, AgentFactory],
    *,
    games: int,
    seed_bases: Sequence[int],
    observation: ObservationFamily,
    first_observation: ObservationFamily | None = None,
    second_observation: ObservationFamily | None = None,
    matchups: Sequence[tuple[str, str]] = PHASE6_MATCHUPS,
    first_name: str = "first",
    second_name: str = "second",
    workers: int = 1,
    checkpoint_paths: (Mapping[str, Path | str] | Sequence[Path | str] | None) = None,
) -> PairedEvaluation:
    """Compare two policies on identical rotated games and seed blocks.

    When workers > 1, baseline names resolve to default factories and the
    compared policies load from checkpoint_paths; in-memory factories are
    ignored in that case.
    """
    if games < 1:
        raise ValueError("games must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    if len(seed_bases) != len(matchups):
        raise ValueError("one seed base is required per matchup")
    roster = dict(baseline_roster)
    first_family = first_observation or observation
    second_family = second_observation or observation
    if workers == 1:
        games_out: list[PairedGame] = []
        for matchup_index, pair in enumerate(matchups):
            missing = set(pair) - set(roster)
            if missing:
                raise ValueError(f"paired roster is missing agents: {sorted(missing)}")
            for learner_seat in range(3):
                first_lineup = list(pair)
                first_lineup.insert(learner_seat, first_name)
                second_lineup = list(pair)
                second_lineup.insert(learner_seat, second_name)
                first_observations = tuple(
                    first_family
                    if seat == learner_seat
                    else ObservationFamily.DECK_AWARE
                    for seat in range(3)
                )
                second_observations = tuple(
                    second_family
                    if seat == learner_seat
                    else ObservationFamily.DECK_AWARE
                    for seat in range(3)
                )
                for game_index in range(games):
                    seed = (
                        seed_bases[matchup_index] + learner_seat * 100_000 + game_index
                    )
                    first_result = run_game(
                        tuple(
                            first_factory if name == first_name else roster[name]
                            for name in first_lineup
                        ),
                        seed=seed,
                        observation=first_family,
                        agent_observations=first_observations,
                    )
                    second_result = run_game(
                        tuple(
                            second_factory if name == second_name else roster[name]
                            for name in second_lineup
                        ),
                        seed=seed,
                        observation=second_family,
                        agent_observations=second_observations,
                    )
                    games_out.append(
                        PairedGame(
                            seed=seed,
                            block=matchup_index,
                            matchup="/".join(pair),
                            learner_seat=learner_seat,
                            first_win_share=_learner_share(first_result, learner_seat),
                            second_win_share=_learner_share(
                                second_result, learner_seat
                            ),
                        )
                    )
        return PairedEvaluation(tuple(games_out), first_name, second_name)

    from flip7.evaluation.phase7 import heldout_factories
    from flip7.training.ppo import baseline_factories

    known_baselines = set(baseline_factories().keys()) | set(heldout_factories().keys())
    if isinstance(checkpoint_paths, Mapping):
        paths = {k: str(v) for k, v in checkpoint_paths.items()}
    elif isinstance(checkpoint_paths, Sequence):
        paths = {}
        if len(checkpoint_paths) >= 2:
            paths[first_name] = str(checkpoint_paths[0])
            paths[second_name] = str(checkpoint_paths[1])
        elif len(checkpoint_paths) == 1:
            paths[first_name] = str(checkpoint_paths[0])
    else:
        paths = {}
    if first_name not in known_baselines and first_name not in paths:
        raise ValueError(
            f"checkpoint_path is required for '{first_name}' when workers > 1"
        )
    if second_name not in known_baselines and second_name not in paths:
        raise ValueError(
            f"checkpoint_path is required for '{second_name}' when workers > 1"
        )

    scheduled_pairs: list[ScheduledPairedGame] = []
    for matchup_index, pair in enumerate(matchups):
        missing = set(pair) - set(roster)
        if missing:
            raise ValueError(f"paired roster is missing agents: {sorted(missing)}")
        for learner_seat in range(3):
            first_lineup = list(pair)
            first_lineup.insert(learner_seat, first_name)
            second_lineup = list(pair)
            second_lineup.insert(learner_seat, second_name)
            first_observations = tuple(
                first_family if seat == learner_seat else ObservationFamily.DECK_AWARE
                for seat in range(3)
            )
            second_observations = tuple(
                second_family if seat == learner_seat else ObservationFamily.DECK_AWARE
                for seat in range(3)
            )
            for game_index in range(games):
                seed = seed_bases[matchup_index] + learner_seat * 100_000 + game_index
                scheduled_pairs.append(
                    ScheduledPairedGame(
                        matchup_index=matchup_index,
                        learner_seat=learner_seat,
                        game_index=game_index,
                        seed=seed,
                        matchup="/".join(pair),
                        first_lineup=tuple(first_lineup),
                        second_lineup=tuple(second_lineup),
                        first_observations=first_observations,
                        second_observations=second_observations,
                        first_family=first_family,
                        second_family=second_family,
                    )
                )

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_paired_worker,
        initargs=(tuple(paths.items()),),
    ) as executor:
        games_out = list(
            executor.map(_run_paired_game_in_worker, scheduled_pairs, chunksize=4)
        )

    return PairedEvaluation(tuple(games_out), first_name, second_name)


def _learner_share(result: GameResult, seat: int) -> float:
    winners = set(result.winning_player_ids)
    return 1.0 / len(winners) if winners and seat in winners else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _normal_interval(values: Sequence[float]) -> list[float]:
    """Return a deterministic normal-approximation interval for diagnostics."""
    if not values:
        return [0.0, 0.0]
    mean = _mean(values)
    if len(values) == 1:
        return [mean, mean]
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    margin = 1.96 * math.sqrt(variance / len(values))
    return [mean - margin, mean + margin]


__all__ = [
    "PairedEvaluation",
    "PairedGame",
    "run_paired_rotated_evaluation",
]
