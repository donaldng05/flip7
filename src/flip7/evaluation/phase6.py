"""Seat-rotated evaluation and aggregation for Phase 6 experiments."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from flip7.agents import AgentFactory
from flip7.envs import ObservationFamily
from flip7.evaluation.metrics import MatchupMetrics
from flip7.evaluation.tournament import run_matchup

type MatchupPair = tuple[str, str]

PHASE6_MATCHUPS: tuple[MatchupPair, ...] = (
    ("random", "threshold"),
    ("risk", "ev"),
    ("dp", "threshold"),
)
PHASE6_SEED_BASES: tuple[int, ...] = (10_000, 11_000, 12_000)


def run_rotated_matchups(
    ppo_factory: AgentFactory,
    baseline_roster: Mapping[str, AgentFactory],
    *,
    games: int = 100,
    seed_bases: Sequence[int] = PHASE6_SEED_BASES,
    observation: ObservationFamily = ObservationFamily.DECK_AWARE,
    matchups: Sequence[MatchupPair] = PHASE6_MATCHUPS,
) -> list[MatchupMetrics]:
    """Evaluate one PPO policy in every seat for each Phase 6 matchup."""
    if len(seed_bases) != len(matchups):
        raise ValueError("one evaluation seed base is required per matchup")
    roster = dict(baseline_roster) | {"ppo": ppo_factory}
    results: list[MatchupMetrics] = []
    for matchup_index, pair in enumerate(matchups):
        if len(pair) != 2:
            raise ValueError("Phase 6 matchups must contain exactly two opponents")
        for learner_seat in range(3):
            lineup = list(pair)
            lineup.insert(learner_seat, "ppo")
            agent_observations = tuple(
                observation if seat == learner_seat else ObservationFamily.DECK_AWARE
                for seat in range(3)
            )
            results.append(
                run_matchup(
                    tuple(lineup),
                    roster,
                    games=games,
                    seed=seed_bases[matchup_index],
                    observation=observation,
                    agent_observations=agent_observations,
                )
            )
    return results


def _confidence_interval(win_share: float, games: int) -> list[float]:
    if games < 1:
        return [0.0, 0.0]
    margin = 1.96 * math.sqrt(max(0.0, win_share * (1.0 - win_share)) / games)
    return [max(0.0, win_share - margin), min(1.0, win_share + margin)]


def _learner_summary(results: Sequence[MatchupMetrics]) -> dict[str, object]:
    games = sum(result.games for result in results)
    if not games:
        raise ValueError("cannot summarize an empty evaluation")
    win_share = sum(result.win_share.get("ppo", 0.0) for result in results) / games
    final_score = sum(result.final_score.get("ppo", 0.0) for result in results) / games
    round_score = sum(result.round_score.get("ppo", 0.0) for result in results) / games
    bust_rate = sum(result.bust_rate.get("ppo", 0.0) for result in results) / games
    action_counts: Counter[str] = Counter()
    for result in results:
        action_counts.update(result.action_counts.get("ppo", Counter()))
    action_total = sum(action_counts.values())
    action_share = {
        action: count / action_total for action, count in sorted(action_counts.items())
    }
    return {
        "games": games,
        "win_share": win_share,
        "approximate_95_ci": _confidence_interval(win_share, games),
        "average_final_score": final_score,
        "average_round_score": round_score,
        "bust_rate": bust_rate,
        "flip7_frequency": sum(
            result.flip7_frequency * result.games for result in results
        )
        / games,
        "tie_frequency": sum(result.tie_frequency * result.games for result in results)
        / games,
        "action_counts": dict(sorted(action_counts.items())),
        "action_share": action_share,
    }


def summarize_rotated_results(
    results: Sequence[MatchupMetrics],
) -> dict[str, object]:
    """Aggregate PPO metrics by seat and matchup across rotated results."""
    if not results:
        raise ValueError("cannot summarize an empty evaluation")

    per_seat: dict[str, dict[str, object]] = {}
    for seat in range(3):
        seat_results = [
            result for result in results if result.agents.index("ppo") == seat
        ]
        if not seat_results:
            raise ValueError(f"evaluation is missing learner seat {seat}")
        per_seat[str(seat)] = _learner_summary(seat_results)

    per_matchup: dict[str, dict[str, object]] = {}
    for result in results:
        opponents = tuple(name for name in result.agents if name != "ppo")
        matchup_name = "/".join(opponents)
        per_matchup.setdefault(matchup_name, {})
    for matchup_name in per_matchup:
        matchup_results = [
            result
            for result in results
            if "/".join(name for name in result.agents if name != "ppo") == matchup_name
        ]
        per_matchup[matchup_name] = _learner_summary(matchup_results)

    all_results = _learner_summary(results)
    seat_values = [
        float(cast(float, per_seat[str(seat)]["win_share"])) for seat in range(3)
    ]
    all_results["seat_spread"] = max(seat_values) - min(seat_values)
    all_results["per_seat"] = per_seat
    all_results["per_matchup"] = per_matchup
    return all_results


def write_phase6_results(
    path: Path,
    metadata: Mapping[str, object],
    results: Sequence[MatchupMetrics],
) -> None:
    """Write Phase 6 metadata, matchup metrics, and PPO summaries as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = dict(metadata)
    payload["matchups"] = [result.as_dict() for result in results]
    payload["summary"] = summarize_rotated_results(results)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


__all__ = [
    "PHASE6_MATCHUPS",
    "PHASE6_SEED_BASES",
    "run_rotated_matchups",
    "summarize_rotated_results",
    "write_phase6_results",
]
