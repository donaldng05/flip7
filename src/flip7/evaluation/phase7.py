"""Round-robin tournaments and Elo aggregation for Phase 7 leagues."""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from flip7.agents import (
    AgentFactory,
    BustProbabilityAgent,
    ExpectedValueAgent,
    FixedThresholdAgent,
    RoundDPAgent,
)
from flip7.envs import ObservationFamily, RewardMode
from flip7.evaluation.metrics import GameResult, MatchupMetrics
from flip7.evaluation.tournament import run_game


@dataclass(frozen=True, slots=True)
class TournamentParticipant:
    """Named policy and the observation family it expects."""

    name: str
    factory: AgentFactory
    observation: ObservationFamily


@dataclass
class EloTable:
    """Multi-player Elo ratings updated through pairwise game outcomes."""

    names: tuple[str, ...]
    initial_rating: float = 1500.0
    k_factor: float = 32.0
    ratings: dict[str, float] = field(init=False)
    games: dict[str, int] = field(init=False)
    win_shares: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        if len(self.names) < 2 or len(set(self.names)) != len(self.names):
            raise ValueError("Elo table requires at least two unique names")
        if not math.isfinite(self.initial_rating):
            raise ValueError("initial_rating must be finite")
        if self.k_factor <= 0 or not math.isfinite(self.k_factor):
            raise ValueError("k_factor must be positive and finite")
        self.ratings = {name: self.initial_rating for name in self.names}
        self.games = dict.fromkeys(self.names, 0)
        self.win_shares = dict.fromkeys(self.names, 0.0)

    def update(self, agents: Sequence[str], winning_player_ids: Sequence[int]) -> None:
        """Apply one simultaneous pairwise update for a multi-player game."""
        if len(agents) < 2 or len(set(agents)) != len(agents):
            raise ValueError("an Elo game requires unique agent names")
        if any(name not in self.ratings for name in agents):
            raise ValueError("Elo game contains an unknown agent")
        winners = set(winning_player_ids)
        if not winners or any(index < 0 or index >= len(agents) for index in winners):
            raise ValueError("Elo game must contain valid winners")

        before = {name: self.ratings[name] for name in agents}
        deltas = dict.fromkeys(agents, 0.0)
        pair_scale = self.k_factor / (len(agents) - 1)
        for first, second in itertools.combinations(range(len(agents)), 2):
            first_name = agents[first]
            second_name = agents[second]
            expected = 1.0 / (
                1.0 + 10.0 ** ((before[second_name] - before[first_name]) / 400.0)
            )
            if first in winners and second in winners:
                actual = 0.5
            elif first in winners:
                actual = 1.0
            elif second in winners:
                actual = 0.0
            else:
                actual = 0.5
            delta = pair_scale * (actual - expected)
            deltas[first_name] += delta
            deltas[second_name] -= delta

        for name, delta in deltas.items():
            self.ratings[name] += delta
        share = 1.0 / len(winners)
        for seat, name in enumerate(agents):
            self.games[name] += 1
            if seat in winners:
                self.win_shares[name] += share

    def as_dict(self) -> dict[str, object]:
        """Return stable rating rows sorted from strongest to weakest."""
        rows = [
            {
                "name": name,
                "rating": self.ratings[name],
                "games": self.games[name],
                "win_share": (
                    self.win_shares[name] / self.games[name]
                    if self.games[name]
                    else 0.0
                ),
            }
            for name in self.names
        ]
        rows.sort(key=lambda row: (-float(row["rating"]), str(row["name"])))
        return {
            "initial_rating": self.initial_rating,
            "k_factor": self.k_factor,
            "ratings": rows,
        }


@dataclass(frozen=True, slots=True)
class TournamentResult:
    """Complete tournament metrics and final Elo ratings."""

    games: int
    lineup_games: int
    matchups: tuple[MatchupMetrics, ...]
    elo: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible tournament results."""
        return {
            "games": self.games,
            "lineup_games": self.lineup_games,
            "matchups": [matchup.as_dict() for matchup in self.matchups],
            "elo": self.elo,
        }


def heldout_factories() -> dict[str, AgentFactory]:
    """Return parameter variants excluded from the Phase 7 training roster."""
    return {
        "threshold_high": lambda: FixedThresholdAgent(threshold=22.0),
        "risk_strict": lambda: BustProbabilityAgent(risk_tolerance=0.10),
        "ev_penalized": lambda: ExpectedValueAgent(bust_penalty=5.0),
        "dp_reduced": lambda: RoundDPAgent(max_unique_numbers=6),
    }


def run_round_robin(
    participants: Sequence[TournamentParticipant],
    *,
    games: int = 10,
    seed: int = 20_000,
    reward: RewardMode = RewardMode.SPARSE_WIN,
    initial_rating: float = 1500.0,
    k_factor: float = 32.0,
) -> TournamentResult:
    """Run every three-player lineup with all six seat permutations."""
    if len(participants) < 3:
        raise ValueError("a tournament requires at least three participants")
    names = tuple(participant.name for participant in participants)
    if len(set(names)) != len(names):
        raise ValueError("tournament participant names must be unique")
    if games < 1:
        raise ValueError("games must be positive")

    elo = EloTable(names, initial_rating=initial_rating, k_factor=k_factor)
    matchup_results: list[MatchupMetrics] = []
    total_games = 0
    lineup_games = games * math.factorial(3)
    for lineup_index, lineup in enumerate(itertools.combinations(participants, 3)):
        canonical_names = tuple(participant.name for participant in lineup)
        metrics = MatchupMetrics(canonical_names, 3)
        for game_index in range(games):
            game_seed = seed + lineup_index * 1_000_000 + game_index
            for permutation in itertools.permutations(lineup):
                observations = tuple(
                    participant.observation for participant in permutation
                )
                result = run_game(
                    tuple(participant.factory for participant in permutation),
                    seed=game_seed,
                    observation=ObservationFamily.DECK_AWARE,
                    reward=reward,
                    agent_observations=observations,
                )
                named_result = GameResult(
                    agents=tuple(participant.name for participant in permutation),
                    player_count=result.player_count,
                    winning_player_ids=result.winning_player_ids,
                    final_scores=result.final_scores,
                    round_scores=result.round_scores,
                    rounds=result.rounds,
                    busted_rounds=result.busted_rounds,
                    flip7_events=result.flip7_events,
                    action_counts=result.action_counts,
                )
                metrics.add(named_result)
                elo.update(named_result.agents, named_result.winning_player_ids)
                total_games += 1
        matchup_results.append(metrics)

    return TournamentResult(
        games=total_games,
        lineup_games=lineup_games,
        matchups=tuple(matchup_results),
        elo=elo.as_dict(),
    )


def write_tournament(path: Path, result: TournamentResult) -> None:
    """Write deterministic tournament metrics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "EloTable",
    "TournamentParticipant",
    "TournamentResult",
    "heldout_factories",
    "run_round_robin",
    "write_tournament",
]
