"""Reproducible AEC tournament execution for baseline agents."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from flip7.agents import ActionMask, Agent, AgentFactory, Observation
from flip7.envs import Flip7AECEnv, ObservationFamily, RewardMode, agent_name
from flip7.envs.observations import encode_observation
from flip7.evaluation.metrics import GameResult, MatchupMetrics

type LastResult = tuple[
    Observation | None,
    float,
    bool,
    bool,
    dict[str, ActionMask],
]


def load_worker_policy(checkpoint_path: Path) -> Agent:
    """Load a policy agent from a checkpoint path inside a worker process."""
    import torch

    payload: dict[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if "actor_state" in payload:
        from flip7.training.mappo import MAPPOAgent

        return MAPPOAgent.from_checkpoint(
            checkpoint_path, deterministic=True, device="cpu"
        )
    from flip7.agents import PPOAgent

    return PPOAgent.from_checkpoint(checkpoint_path, deterministic=True, device="cpu")


@dataclass(frozen=True, slots=True)
class ScheduledMatchupGame:
    """One scheduled matchup game to be executed in a worker process."""

    game_index: int
    seed: int
    names: tuple[str, ...]
    observation: ObservationFamily
    agent_observations: tuple[ObservationFamily, ...] | None


_worker_roster: dict[str, AgentFactory] = {}


def _initialize_matchup_worker(
    checkpoint_paths: tuple[tuple[str, str], ...],
) -> None:
    """Initialize worker process with thread count and cached policies."""
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


def _run_matchup_game_in_worker(item: ScheduledMatchupGame) -> GameResult:
    """Execute one scheduled matchup game in a worker process."""
    factories = tuple(_worker_roster[name] for name in item.names)
    result = run_game(
        factories,
        seed=item.seed,
        observation=item.observation,
        agent_observations=item.agent_observations,
    )
    return GameResult(
        agents=item.names,
        player_count=result.player_count,
        winning_player_ids=result.winning_player_ids,
        final_scores=result.final_scores,
        round_scores=result.round_scores,
        rounds=result.rounds,
        busted_rounds=result.busted_rounds,
        flip7_events=result.flip7_events,
        action_counts=result.action_counts,
    )


def run_game(
    factories: Sequence[AgentFactory],
    *,
    seed: int,
    observation: ObservationFamily = ObservationFamily.DECK_AWARE,
    reward: RewardMode = RewardMode.SPARSE_WIN,
    agent_observations: Sequence[ObservationFamily] | None = None,
) -> GameResult:
    """Run one seeded complete game for the supplied seat lineup."""
    player_count = len(factories)
    if agent_observations is not None and len(agent_observations) != player_count:
        raise ValueError("agent_observations must match the lineup length")
    env_observation = (
        ObservationFamily.DECK_AWARE if agent_observations is not None else observation
    )
    env = Flip7AECEnv(player_count, observation=env_observation, reward=reward)
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
        if agent_observations is not None:
            observation_value = encode_observation(
                env.engine.state,
                seat,
                agent_observations[seat],
            )
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
    agent_observations: Sequence[ObservationFamily] | None = None,
    workers: int = 1,
    checkpoint_paths: Mapping[str, Path | str] | None = None,
) -> MatchupMetrics:
    """Run repeated games for one explicit seat lineup."""
    if len(names) < 3:
        raise ValueError("a matchup requires at least three seats")
    if player_count is not None and player_count != len(names):
        raise ValueError("player_count must match the lineup length")
    if games < 1:
        raise ValueError("games must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    if agent_observations is not None and len(agent_observations) != len(names):
        raise ValueError("agent_observations must match the lineup length")
    missing = set(names) - set(roster)
    if missing:
        raise ValueError(f"roster is missing agents: {sorted(missing)}")
    metrics = MatchupMetrics(tuple(names), len(names))
    if workers == 1:
        for game_index in range(games):
            factories = tuple(roster[name] for name in names)
            result = run_game(
                factories,
                seed=seed + game_index,
                observation=observation,
                agent_observations=agent_observations,
            )
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

    from flip7.evaluation.phase7 import heldout_factories
    from flip7.training.ppo import baseline_factories

    known_baselines = set(baseline_factories().keys()) | set(heldout_factories().keys())
    paths = {k: str(v) for k, v in (checkpoint_paths or {}).items()}
    for name in names:
        if name not in known_baselines and name not in paths:
            raise ValueError(
                f"checkpoint_path is required for agent '{name}' when workers > 1"
            )

    scheduled = [
        ScheduledMatchupGame(
            game_index=game_index,
            seed=seed + game_index,
            names=tuple(names),
            observation=observation,
            agent_observations=(
                tuple(agent_observations) if agent_observations is not None else None
            ),
        )
        for game_index in range(games)
    ]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_matchup_worker,
        initargs=(tuple(paths.items()),),
    ) as executor:
        for result in executor.map(_run_matchup_game_in_worker, scheduled, chunksize=4):
            metrics.add(result)
    return metrics


def write_results(path: Path, results: Sequence[MatchupMetrics]) -> None:
    """Write JSON evaluation results with stable, human-readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"matchups": [result.as_dict() for result in results]}
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
