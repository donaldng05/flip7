"""Standard rotated baseline and held-out evaluation runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from flip7.agents import Agent
from flip7.envs import ObservationFamily
from flip7.evaluation import (
    heldout_factories,
    run_rotated_matchups,
    summarize_rotated_results,
)
from flip7.experiment.parsing import as_ints, as_mapping, as_pairs
from flip7.experiment.policies import cached_policy
from flip7.training import baseline_factories


def run_standard_evaluations(
    root: Mapping[str, object],
    checkpoint: Path,
    *,
    games: int,
    seed_offset: int,
    observation: ObservationFamily,
    policy_factory: Callable[[], Agent] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Run rotated baseline and held-out evaluation matchups for a checkpoint."""
    evaluation = as_mapping(root["evaluation"], "evaluation")
    seed_bases = as_ints(evaluation["seed_bases"], "evaluation.seed_bases")
    heldout_bases = as_ints(
        evaluation["heldout_seed_bases"], "evaluation.heldout_seed_bases"
    )
    matchups = as_pairs(evaluation["matchups"], "evaluation.matchups")
    heldout_matchups = as_pairs(
        evaluation["heldout_matchups"], "evaluation.heldout_matchups"
    )
    policy = cached_policy(checkpoint) if policy_factory is None else policy_factory
    baseline_results = run_rotated_matchups(
        policy,
        baseline_factories(),
        games=games,
        seed_bases=tuple(base + seed_offset for base in seed_bases),
        observation=observation,
        matchups=matchups,
    )
    heldout_results = run_rotated_matchups(
        policy,
        baseline_factories() | heldout_factories(),
        games=games,
        seed_bases=tuple(base + seed_offset for base in heldout_bases),
        observation=observation,
        matchups=heldout_matchups,
    )
    return (
        {
            "matchups": [result.as_dict() for result in baseline_results],
            "summary": summarize_rotated_results(baseline_results),
        },
        {
            "matchups": [result.as_dict() for result in heldout_results],
            "summary": summarize_rotated_results(heldout_results),
        },
    )
