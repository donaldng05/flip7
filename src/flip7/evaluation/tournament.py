"""Reproducible AEC tournament execution for baseline agents."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from flip7.agents import ActionMask, AgentFactory, Observation
from flip7.envs import Flip7AECEnv, ObservationFamily, RewardMode, agent_name
from flip7.evaluation.metrics import GameResult, MatchupMetrics

type LastResult = tuple[
    Observation | None,
    float,
    bool,
    bool,
    dict[str, ActionMask],
]


def run_game(
    factories: Sequence[AgentFactory],
    *,
    seed: int,
    observation: ObservationFamily = ObservationFamily.DECK_AWARE,
    reward: RewardMode = RewardMode.SPARSE_WIN,
) -> GameResult:
    """Run one seeded complete game for the supplied seat lineup."""
    player_count = len(factories)
    env = Flip7AECEnv(player_count, observation=observation, reward=reward)
    policies = {agent_name(seat): factory() for seat, factory in enumerate(factories)}
    env.reset(seed=seed)
    action_counts = [{"hit": 0, "stay": 0, "target": 0} for _ in range(player_count)]
    previous_scores = [0] * player_count
    round_scores = [0] * player_count
    busted_seats: set[tuple[int, int]] = set()
    flip7_events: set[tuple[int, int]] = set()

    for _ in env.agent_iter(max_iter=100_000):
        observation_value, _reward, terminated, truncated, info = cast(
            LastResult, env.last()
        )
        if terminated or truncated:
            env.step(None)
            continue
        if observation_value is None:
            raise RuntimeError("active AEC agent has no observation")
        seat = int(env.agent_selection.rsplit("_", 1)[1])
        action = policies[env.agent_selection](observation_value, info["action_mask"])
        if action == 0:
            action_counts[seat]["hit"] += 1
        elif action == 1:
            action_counts[seat]["stay"] += 1
        else:
            action_counts[seat]["target"] += 1
        env.step(action)
        state = env.engine.state
        for player_id, player in enumerate(state.players):
            delta = player.cumulative_score - previous_scores[player_id]
            if delta > 0:
                round_scores[player_id] += delta
            previous_scores[player_id] = player.cumulative_score
            if player.status.value == "busted":
                busted_seats.add((state.round_number, player_id))
        if state.flip7_player_id is not None:
            flip7_events.add((state.round_number, state.flip7_player_id))

    state = env.engine.state
    return GameResult(
        agents=tuple(str(seat) for seat in range(player_count)),
        player_count=player_count,
        winning_player_ids=state.winning_player_ids,
        final_scores=tuple(player.cumulative_score for player in state.players),
        round_scores=tuple(round_scores),
        rounds=state.round_number,
        busted_rounds=tuple(
            sum(player_id == seat for _round, player_id in busted_seats)
            for seat in range(player_count)
        ),
        flip7_events=len(flip7_events),
        action_counts=tuple(action_counts),
    )


def run_matchup(
    names: Sequence[str],
    roster: Mapping[str, AgentFactory],
    *,
    games: int = 100,
    seed: int = 7,
    player_count: int | None = None,
    observation: ObservationFamily = ObservationFamily.DECK_AWARE,
) -> MatchupMetrics:
    """Run repeated games for one explicit seat lineup."""
    if len(names) < 3:
        raise ValueError("a matchup requires at least three seats")
    if player_count is not None and player_count != len(names):
        raise ValueError("player_count must match the lineup length")
    if games < 1:
        raise ValueError("games must be positive")
    missing = set(names) - set(roster)
    if missing:
        raise ValueError(f"roster is missing agents: {sorted(missing)}")
    metrics = MatchupMetrics(tuple(names), len(names))
    for game_index in range(games):
        factories = tuple(roster[name] for name in names)
        result = run_game(factories, seed=seed + game_index, observation=observation)
        result = GameResult(
            agents=tuple(names),
            player_count=result.player_count,
            winning_player_ids=result.winning_player_ids,
            final_scores=result.final_scores,
            round_scores=result.round_scores,
            rounds=result.rounds,
            busted_rounds=result.busted_rounds,
            flip7_events=result.flip7_events,
            action_counts=result.action_counts,
        )
        metrics.add(result)
    return metrics


def write_results(path: Path, results: Sequence[MatchupMetrics]) -> None:
    """Write JSON evaluation results with stable, human-readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"matchups": [result.as_dict() for result in results]}
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
