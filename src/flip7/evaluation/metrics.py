"""Metric models and aggregation for baseline evaluations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


def _float_map() -> dict[str, float]:
    return {}


def _counter_map() -> dict[str, Counter[str]]:
    return {}


def _sample_map() -> dict[str, list[float]]:
    return {}


@dataclass(frozen=True, slots=True)
class GameResult:
    """Public metrics collected from one complete game."""

    agents: tuple[str, ...]
    player_count: int
    winning_player_ids: tuple[int, ...]
    final_scores: tuple[int, ...]
    round_scores: tuple[int, ...]
    rounds: int
    busted_rounds: tuple[int, ...]
    flip7_events: int
    action_counts: tuple[dict[str, int], ...]

    @property
    def tie(self) -> bool:
        return len(self.winning_player_ids) > 1


@dataclass
class MatchupMetrics:
    """Aggregated metrics for a fixed lineup and player count."""

    agents: tuple[str, ...]
    player_count: int
    games: int = 0
    win_share: dict[str, float] = field(default_factory=_float_map)
    final_score: dict[str, float] = field(default_factory=_float_map)
    round_score: dict[str, float] = field(default_factory=_float_map)
    bust_rate: dict[str, float] = field(default_factory=_float_map)
    flip7_frequency: float = 0.0
    tie_frequency: float = 0.0
    action_counts: dict[str, Counter[str]] = field(default_factory=_counter_map)
    # Kept in memory for uncertainty-aware diagnostics.  It is intentionally
    # not serialized by ``as_dict`` so existing evaluation artifacts remain
    # compact and backward-compatible.
    win_share_samples: dict[str, list[float]] = field(default_factory=_sample_map)

    def add(self, result: GameResult) -> None:
        self.games += 1
        winners = set(result.winning_player_ids)
        share = 1.0 / len(winners) if winners else 0.0
        for seat, agent_name in enumerate(result.agents):
            self.win_share[agent_name] = self.win_share.get(agent_name, 0.0) + (
                share if seat in winners else 0.0
            )
            self.win_share_samples.setdefault(agent_name, []).append(
                share if seat in winners else 0.0
            )
            self.final_score[agent_name] = (
                self.final_score.get(agent_name, 0.0) + (result.final_scores[seat])
            )
            self.round_score[agent_name] = (
                self.round_score.get(agent_name, 0.0) + (result.round_scores[seat])
            )
            denominator = max(1, result.rounds)
            self.bust_rate[agent_name] = self.bust_rate.get(agent_name, 0.0) + (
                result.busted_rounds[seat] / denominator
            )
            counts = self.action_counts.setdefault(agent_name, Counter())
            for action, count in result.action_counts[seat].items():
                counts[action] += count
        self.flip7_frequency += float(result.flip7_events > 0)
        self.tie_frequency += float(result.tie)

    def mean(self, values: dict[str, float]) -> dict[str, float]:
        if not self.games:
            return {name: 0.0 for name in self.agents}
        return {name: values[name] / self.games for name in self.agents}

    def as_dict(self) -> dict[str, object]:
        return {
            "agents": list(self.agents),
            "player_count": self.player_count,
            "games": self.games,
            "win_share": self.mean(self.win_share),
            "average_final_score": self.mean(self.final_score),
            "average_round_score": self.mean(self.round_score),
            "bust_rate": self.mean(self.bust_rate),
            "flip7_frequency": self.flip7_frequency / self.games if self.games else 0.0,
            "tie_frequency": self.tie_frequency / self.games if self.games else 0.0,
            "action_counts": {
                name: dict(counts) for name, counts in self.action_counts.items()
            },
        }
