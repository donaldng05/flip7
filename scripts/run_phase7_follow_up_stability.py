"""Run the Phase 7 stability and diversity-benefit follow-up."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from flip7 import __version__
from flip7.agents import PPOAgent
from flip7.config.loader import load_config
from flip7.envs import ObservationFamily
from flip7.evaluation import (
    analyze_snapshots,
    build_paired_schedule,
    direct_final_warmup_comparison,
    heldout_factories,
    matchup_win_share_matrix,
    non_transitive_cycles,
    run_paired_rotated_evaluation,
    run_paired_round_robin,
    run_rotated_matchups,
    schedule_as_dict,
    summarize_rotated_results,
    validate_followup_manifest,
)
from flip7.evaluation.phase7 import TournamentParticipant
from flip7.training import (
    FollowUpLeagueConfig,
    MAPPOAgent,
    MAPPOConfig,
    MAPPOTrainer,
    StabilityControlPPOTrainer,
    StabilityLeaguePPOTrainer,
    baseline_factories,
    write_history,
    write_mappo_history,
)
from flip7.training.ppo import PPOConfig


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _ints(value: object, name: str) -> tuple[int, ...]:
    values = _list(value, name)
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in values):
        raise ValueError(f"{name} must contain only integers")
    return tuple(cast(int, item) for item in values)


def _strings(value: object, name: str) -> tuple[str, ...]:
    values = _list(value, name)
    if not all(isinstance(item, str) for item in values):
        raise ValueError(f"{name} must contain only strings")
    return tuple(cast(str, item) for item in values)


def _pairs(value: object, name: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for index, item in enumerate(_list(value, name)):
        pair = _strings(item, f"{name}[{index}]")
        if len(pair) != 2:
            raise ValueError(f"{name} entries must contain two names")
        pairs.append((pair[0], pair[1]))
    if not pairs:
        raise ValueError(f"{name} must not be empty")
    return tuple(pairs)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_config(
    root: Mapping[str, object],
    seed: int,
    *,
    updates: int,
    observation_override: str | None = None,
) -> PPOConfig:
    environment = _mapping(root["env"], "env")
    training = dict(_mapping(root["training"], "training"))
    training.pop("algorithm", None)
    training.update(
        {
            "seed": seed,
            "updates": updates,
            "player_count": int(root["players"]),
            "learner_id": int(environment["learner_id"]),
            "learner_seat_mode": str(environment["learner_seat_mode"]),
            "observation": observation_override or str(environment["observation"]),
            "reward": str(environment["reward"]),
        }
    )
    return PPOConfig(**training)


def _league_config(
    root: Mapping[str, object], condition: Mapping[str, object]
) -> FollowUpLeagueConfig:
    defaults = dict(_mapping(root["population_defaults"], "population_defaults"))
    if "population" in condition:
        defaults.update(_mapping(condition["population"], "condition.population"))
    opponents = _mapping(root["opponents"], "opponents")
    defaults["baseline_names"] = _strings(opponents["training"], "opponents.training")
    return FollowUpLeagueConfig(**defaults)


def _cached_policy(path: Path) -> Any:
    policy: PPOAgent | None = None

    def factory() -> PPOAgent:
        nonlocal policy
        if policy is None:
            policy = PPOAgent.from_checkpoint(path, deterministic=True)
        return policy

    return factory


def _participants(
    checkpoint: Path,
    observation: ObservationFamily,
    snapshots: Sequence[Any],
    warmup: Any | None,
    *,
    include_snapshots: bool,
) -> tuple[TournamentParticipant, ...]:
    participants = [
        TournamentParticipant(name, factory, ObservationFamily.DECK_AWARE)
        for name, factory in baseline_factories().items()
    ]
    participants.append(
        TournamentParticipant("final", _cached_policy(checkpoint), observation)
    )
    if include_snapshots:
        participants.extend(
            TournamentParticipant(
                snapshot.policy_id,
                _cached_policy(snapshot.path),
                snapshot.observation,
            )
            for snapshot in snapshots
        )
    if warmup is not None and all(
        participant.name != warmup.policy_id for participant in participants
    ):
        participants.append(
            TournamentParticipant(
                warmup.policy_id,
                _cached_policy(warmup.path),
                warmup.observation,
            )
        )
    return tuple(participants)


def _evaluate(
    root: Mapping[str, object],
    checkpoint: Path,
    *,
    games: int,
    seed_offset: int,
    observation: ObservationFamily,
    policy_factory: Any | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    evaluation = _mapping(root["evaluation"], "evaluation")
    seed_bases = _ints(evaluation["seed_bases"], "evaluation.seed_bases")
    heldout_bases = _ints(
        evaluation["heldout_seed_bases"], "evaluation.heldout_seed_bases"
    )
    matchups = _pairs(evaluation["matchups"], "evaluation.matchups")
    heldout_matchups = _pairs(
        evaluation["heldout_matchups"], "evaluation.heldout_matchups"
    )
    policy = _cached_policy(checkpoint) if policy_factory is None else policy_factory
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


def _mappo_config(
    root: Mapping[str, object], seed: int, *, updates: int
) -> MAPPOConfig:
    environment = _mapping(root["env"], "env")
    training = dict(_mapping(root["training"], "training"))
    training.pop("algorithm", None)
    training.update(
        {
            "seed": seed,
            "updates": updates,
            "player_count": int(root["players"]),
            "observation": str(environment["observation"]),
            "reward": str(environment["reward"]),
        }
    )
    return MAPPOConfig(**training)


def _run_mappo(
    root: Mapping[str, object],
    output_root: Path,
    *,
    seeds: Sequence[int],
    updates: int,
    games: int,
    focused_games: int,
    workers: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    observation = ObservationFamily(str(_mapping(root["env"], "env")["observation"]))
    tournament_values = _mapping(root["tournament"], "tournament")
    for seed in seeds:
        run_dir = output_root / "mappo" / f"seed-{seed}"
        checkpoint = run_dir / "checkpoint.pt"
        trainer = MAPPOTrainer(_mappo_config(root, seed, updates=updates))
        history = trainer.train(checkpoint)
        write_mappo_history(run_dir / "training.json", history)
        evaluation, heldout = _evaluate(
            root,
            checkpoint,
            games=games,
            seed_offset=seed * 10_000,
            observation=observation,
            policy_factory=lambda path=checkpoint: MAPPOAgent.from_checkpoint(
                path, deterministic=True
            ),
        )
        participants = tuple(
            [
                TournamentParticipant(name, factory, ObservationFamily.DECK_AWARE)
                for name, factory in baseline_factories().items()
            ]
            + [
                TournamentParticipant(
                    "mappo",
                    lambda path=checkpoint: MAPPOAgent.from_checkpoint(
                        path, deterministic=True
                    ),
                    observation,
                )
            ]
        )
        tournament, schedule = run_paired_round_robin(
            participants,
            games=focused_games,
            seed=int(tournament_values["seed"]) + seed,
            initial_rating=float(tournament_values["initial_rating"]),
            k_factor=float(tournament_values["k_factor"]),
            workers=1,
        )
        population_path = run_dir / "population.json"
        diversity_path = run_dir / "diversity.json"
        evaluation_path = run_dir / "evaluation.json"
        tournament_path = run_dir / "tournament.json"
        manifest_path = run_dir / "manifest.json"
        _write_json(
            population_path,
            {
                "algorithm": "mappo",
                "trainable_seats": int(root["players"]),
                "rollout_steps": trainer.config.rollout_steps,
            },
        )
        _write_json(
            diversity_path,
            {
                "algorithm": "mappo",
                "critic_input_size": trainer.critic_input_size,
                "actor_observation_size": trainer.observation_size,
            },
        )
        _write_json(
            evaluation_path,
            {
                "experiment": str(root["experiment"]),
                "condition": "mappo",
                "seed": seed,
                "baseline": evaluation,
                "heldout": heldout,
            },
        )
        _write_json(
            tournament_path,
            tournament.as_dict() | {"schedule": schedule_as_dict(schedule)},
        )
        manifest = {
            "experiment": str(root["experiment"]),
            "condition": "mappo",
            "seed": seed,
            "checkpoint": str(checkpoint),
            "training_history": str(run_dir / "training.json"),
            "population": str(population_path),
            "diversity": str(diversity_path),
            "evaluation": str(evaluation_path),
            "tournament": str(tournament_path),
            "manifest": str(manifest_path),
            "critic_input_size": trainer.critic_input_size,
            "fixed_final_update": updates,
            "tournament_workers": workers,
        }
        _write_json(manifest_path, manifest)
        validate_followup_manifest(manifest)
        rows.append(
            {
                "condition": "mappo",
                "seed": seed,
                "baseline": evaluation["summary"],
                "heldout": heldout["summary"],
                "critic_input_size": trainer.critic_input_size,
                "tournament": tournament.elo,
            }
        )
    return rows


def _population_payload(trainer: object) -> dict[str, object]:
    if isinstance(trainer, StabilityLeaguePPOTrainer):
        return trainer.league.as_dict() | {
            "rollout_schedule": trainer.rollout_schedule,
            "training_recipe": asdict(trainer.config),
        }
    if isinstance(trainer, StabilityControlPPOTrainer):
        return {
            "algorithm": "ppo",
            "snapshots": [],
            "warmup_anchor": None,
            "opponent_exposure": {},
            "rollout_schedule": trainer.rollout_schedule,
            "training_recipe": asdict(trainer.config),
        }
    raise TypeError("unsupported stability trainer")


def _rating_rows(elo: Mapping[str, object]) -> list[Mapping[str, object]]:
    values = elo.get("ratings")
    if not isinstance(values, list):
        raise ValueError("tournament ratings must be a list")
    return [_mapping(value, "rating row") for value in values]


def _rating(elo: Mapping[str, object], name: str) -> float | None:
    for row in _rating_rows(elo):
        if row.get("name") == name:
            return float(row["rating"])
    return None


def _run_condition(
    root: Mapping[str, object],
    condition: Mapping[str, object],
    seed: int,
    output_root: Path,
    *,
    updates: int,
    games: int,
    focused_games: int,
    full_games: int,
    workers: int,
    observation_override: str | None = None,
) -> dict[str, object]:
    name = str(condition["name"])
    run_dir = output_root / name / f"seed-{seed}"
    checkpoint = run_dir / "checkpoint.pt"
    history_path = run_dir / "training.json"
    population_path = run_dir / "population.json"
    diversity_path = run_dir / "diversity.json"
    evaluation_path = run_dir / "evaluation.json"
    tournament_path = run_dir / "tournament.json"
    manifest_path = run_dir / "manifest.json"
    observation_value = observation_override or str(
        _mapping(root["env"], "env")["observation"]
    )
    config = _training_config(
        root, seed, updates=updates, observation_override=observation_value
    )
    baseline_names = _strings(
        _mapping(root["opponents"], "opponents")["training"],
        "opponents.training",
    )
    trainer_name = str(condition.get("trainer", "league"))
    league_config = _league_config(root, condition)
    if trainer_name == "control":
        trainer: Any = StabilityControlPPOTrainer(config, baseline_names)
    elif trainer_name == "league":
        trainer = StabilityLeaguePPOTrainer(config, league_config=league_config)
    else:
        raise ValueError(f"unsupported stability trainer: {trainer_name}")
    history = trainer.train(checkpoint)
    write_history(history_path, history)
    population = _population_payload(trainer)
    _write_json(population_path, population)
    snapshots: Sequence[Any] = ()
    archived: Sequence[Any] = ()
    warmup: Any | None = None
    state_bank: Any | None = None
    response_signatures: Mapping[str, Sequence[float]] = {}
    if isinstance(trainer, StabilityLeaguePPOTrainer):
        snapshots = trainer.league.snapshots
        archived = trainer.league.archived_snapshots
        warmup = trainer.league.warmup_anchor
        state_bank = trainer.league.state_bank
        response_signatures = trainer.league.response_signatures
    evaluation, heldout = _evaluate(
        root,
        checkpoint,
        games=games,
        seed_offset=seed * 10_000,
        observation=ObservationFamily(observation_value),
    )
    tournament_values = _mapping(root["tournament"], "tournament")
    participants = _participants(
        checkpoint,
        ObservationFamily(observation_value),
        snapshots,
        warmup,
        include_snapshots=False,
    )
    focused_schedule = build_paired_schedule(
        tuple(item.name for item in participants),
        games=focused_games,
        seed=int(tournament_values["seed"]) + seed,
    )
    checkpoint_paths = {"final": checkpoint}
    if warmup is not None:
        checkpoint_paths[warmup.policy_id] = warmup.path
    focused, focused_schedule = run_paired_round_robin(
        participants,
        games=focused_games,
        seed=int(tournament_values["seed"]) + seed,
        initial_rating=float(tournament_values["initial_rating"]),
        k_factor=float(tournament_values["k_factor"]),
        schedule=focused_schedule,
        workers=workers,
        checkpoint_paths=checkpoint_paths,
    )
    direct = None
    if warmup is not None:
        final_participant = next(item for item in participants if item.name == "final")
        direct = direct_final_warmup_comparison(
            final_participant,
            next(item for item in participants if item.name == warmup.policy_id),
            tuple(item for item in participants if item.name in set(baseline_names)),
            games=focused_games,
            seed=int(tournament_values["seed"]) + seed + 900_000,
        )
    full_participants = _participants(
        checkpoint,
        ObservationFamily(observation_value),
        snapshots,
        warmup,
        include_snapshots=True,
    )
    full_schedule = build_paired_schedule(
        tuple(item.name for item in full_participants),
        games=full_games,
        seed=int(tournament_values["seed"]) + seed + 1_000_000,
    )
    full_paths = dict(checkpoint_paths)
    full_paths.update({item.policy_id: item.path for item in snapshots})
    full, full_schedule = run_paired_round_robin(
        full_participants,
        games=full_games,
        seed=int(tournament_values["seed"]) + seed + 1_000_000,
        initial_rating=float(tournament_values["initial_rating"]),
        k_factor=float(tournament_values["k_factor"]),
        schedule=full_schedule,
        workers=workers,
        checkpoint_paths=full_paths,
    )
    if state_bank is not None:
        diversity = analyze_snapshots(
            archived,
            state_bank,
            response_signatures=response_signatures,
        )
        active_diversity = analyze_snapshots(
            snapshots,
            state_bank,
            response_signatures=response_signatures,
        )
        leave_one_out: list[dict[str, object]] = []
        for excluded in snapshots:
            remaining = tuple(
                item for item in snapshots if item.policy_id != excluded.policy_id
            )
            remaining_metrics = analyze_snapshots(
                remaining,
                state_bank,
                response_signatures=response_signatures,
            )
            leave_one_out.append(
                {
                    "excluded_policy_id": excluded.policy_id,
                    "remaining_policies": list(remaining_metrics["policies"]),
                    "mean_pairwise_js_divergence": remaining_metrics[
                        "mean_pairwise_js_divergence"
                    ],
                    "mean_pairwise_action_disagreement": remaining_metrics[
                        "mean_pairwise_action_disagreement"
                    ],
                    "mean_pairwise_response_distance": remaining_metrics.get(
                        "mean_pairwise_response_distance", 0.0
                    ),
                }
            )
        diversity["active_population"] = active_diversity
        diversity["leave_one_snapshot_out"] = leave_one_out
        diversity["response_signature_weight"] = league_config.response_signature_weight
        diversity["response_signature_games"] = league_config.response_signature_games
        diversity["state_bank"] = state_bank.as_dict()
        diversity["archived_snapshots"] = [item.as_dict() for item in archived]
        diversity["active_snapshots"] = [item.as_dict() for item in snapshots]
        diversity["training_roster"] = list(baseline_names)
        diversity["matchup_win_share_matrix"] = matchup_win_share_matrix(full)
        diversity["non_transitive_cycles"] = non_transitive_cycles(
            cast(
                Mapping[str, Mapping[str, float]],
                diversity["matchup_win_share_matrix"],
            )
        )
    else:
        diversity = {
            "policies": [],
            "active_population": {"policies": []},
            "training_roster": list(baseline_names),
            "matchup_win_share_matrix": matchup_win_share_matrix(full),
            "non_transitive_cycles": [],
        }
    focused_elo = focused.elo
    final_rating = _rating(focused_elo, "final")
    warmup_rating = _rating(focused_elo, warmup.policy_id) if warmup else None
    diversity["focused_tournament_elo"] = focused_elo
    diversity["full_tournament_elo"] = full.elo
    diversity["final_rating"] = final_rating
    diversity["warmup_rating"] = warmup_rating
    diversity["final_minus_warmup_elo"] = (
        final_rating - warmup_rating
        if final_rating is not None and warmup_rating is not None
        else None
    )
    _write_json(diversity_path, cast(Mapping[str, object], diversity))
    _write_json(
        tournament_path,
        {
            "focused": focused.as_dict(),
            "focused_schedule": schedule_as_dict(focused_schedule),
            "full_population": full.as_dict(),
            "full_schedule": schedule_as_dict(full_schedule),
            "direct_final_warmup": direct,
        },
    )
    _write_json(
        evaluation_path,
        {
            "experiment": str(root["experiment"]),
            "condition": name,
            "seed": seed,
            "updates": updates,
            "baseline": evaluation,
            "heldout": heldout,
        },
    )
    manifest: dict[str, object] = {
        "experiment": str(root["experiment"]),
        "condition": name,
        "seed": seed,
        "package_version": __version__,
        "config": asdict(config),
        "population_config": asdict(league_config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "training_history": str(history_path),
        "population": str(population_path),
        "diversity": str(diversity_path),
        "evaluation": str(evaluation_path),
        "tournament": str(tournament_path),
        "manifest": str(manifest_path),
        "fixed_final_update": updates,
        "focused_schedule_games": len(focused_schedule),
        "full_schedule_games": len(full_schedule),
        "tournament_workers": workers,
        "seat_balanced": True,
        "observation": observation_value,
    }
    _write_json(manifest_path, manifest)
    validate_followup_manifest(manifest)
    return {
        "condition": name,
        "seed": seed,
        "baseline": evaluation["summary"],
        "heldout": heldout["summary"],
        "final_rating": final_rating,
        "warmup_rating": warmup_rating,
        "final_minus_warmup_elo": diversity.get("final_minus_warmup_elo"),
        "active_mean_pairwise_js_divergence": _active_metric(
            diversity, "mean_pairwise_js_divergence"
        ),
        "active_mean_pairwise_action_disagreement": _active_metric(
            diversity, "mean_pairwise_action_disagreement"
        ),
        "focused_tournament": focused.elo,
        "full_tournament": full.elo,
        "direct_final_warmup": direct,
        "observation": observation_value,
    }


def _active_metric(payload: Mapping[str, object], key: str) -> float:
    active = payload.get("active_population")
    if not isinstance(active, dict):
        return 0.0
    return float(active.get(key, 0.0))


def _normal_interval(values: Sequence[float]) -> list[float]:
    if not values:
        return [0.0, 0.0]
    mean = sum(values) / len(values)
    if len(values) == 1:
        return [mean, mean]
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    margin = 1.96 * math.sqrt(variance / len(values))
    return [mean - margin, mean + margin]


def _paired_heldout_comparisons(
    root: Mapping[str, object],
    output_root: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    games: int,
) -> list[dict[str, object]]:
    """Run common-seed held-out comparisons for the diversity gate."""
    latest = {
        int(row["seed"]): row
        for row in rows
        if row.get("condition") == "balanced_latest_only"
    }
    diverse_name = _selected_condition_name(root, rows)
    diverse = {
        int(row["seed"]): row for row in rows if row.get("condition") == diverse_name
    }
    if not latest or not diverse:
        return []
    evaluation = _mapping(root["evaluation"], "evaluation")
    heldout_bases = _ints(
        evaluation["heldout_seed_bases"], "evaluation.heldout_seed_bases"
    )
    matchups = _pairs(evaluation["heldout_matchups"], "evaluation.heldout_matchups")
    latest_observation = ObservationFamily(
        str(_mapping(root["env"], "env")["observation"])
    )
    comparisons: list[dict[str, object]] = []
    for seed, diverse_row in sorted(diverse.items()):
        if seed not in latest:
            continue
        latest_path = (
            output_root / "balanced_latest_only" / f"seed-{seed}" / "checkpoint.pt"
        )
        diverse_path = output_root / diverse_name / f"seed-{seed}" / "checkpoint.pt"
        diverse_observation = ObservationFamily(str(diverse_row["observation"]))
        paired = run_paired_rotated_evaluation(
            _cached_policy(diverse_path),
            _cached_policy(latest_path),
            baseline_factories() | heldout_factories(),
            games=games,
            seed_bases=tuple(base + seed * 10_000 for base in heldout_bases),
            observation=latest_observation,
            first_observation=diverse_observation,
            second_observation=latest_observation,
            matchups=matchups,
            first_name="response_diverse",
            second_name="latest_only",
        )
        comparisons.append({"seed": seed, **paired.as_dict()})
    _write_json(
        output_root / "paired-heldout-comparisons.json",
        {"comparisons": comparisons},
    )
    return comparisons


def _aggregate(
    root: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    paired: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        if isinstance(row.get("baseline"), dict) and isinstance(
            row.get("heldout"), dict
        ):
            grouped.setdefault(str(row["condition"]), []).append(row)
    aggregates: dict[str, object] = {}
    for condition, condition_rows in sorted(grouped.items()):
        baseline = [
            cast(Mapping[str, object], row["baseline"]) for row in condition_rows
        ]
        heldout = [cast(Mapping[str, object], row["heldout"]) for row in condition_rows]
        aggregates[condition] = {
            "seeds": [int(row["seed"]) for row in condition_rows],
            "baseline_win_share_mean": sum(
                float(item["win_share"]) for item in baseline
            )
            / len(baseline),
            "baseline_seat_spread": [float(item["seat_spread"]) for item in baseline],
            "heldout_win_share_mean": sum(float(item["win_share"]) for item in heldout)
            / len(heldout),
            "heldout_seat_spread": [float(item["seat_spread"]) for item in heldout],
            "active_mean_pairwise_js_divergence": [
                row.get("active_mean_pairwise_js_divergence") for row in condition_rows
            ],
            "active_mean_pairwise_action_disagreement": [
                row.get("active_mean_pairwise_action_disagreement")
                for row in condition_rows
            ],
            "final_minus_warmup_elo": [
                row.get("final_minus_warmup_elo") for row in condition_rows
            ],
        }

    latest_name = "balanced_latest_only"
    diverse_name = _selected_condition_name(root, rows)
    latest = {int(row["seed"]): row for row in grouped.get(latest_name, [])}
    diverse = {int(row["seed"]): row for row in grouped.get(diverse_name, [])}
    paired_by_seed = {int(item["seed"]): item for item in paired}
    gate_values = _mapping(root["gates"], "gates")
    differences = (
        [float(item["mean_difference"]) for item in paired]
        if paired
        else [
            float(cast(Mapping[str, object], row["heldout"])["win_share"])
            - float(cast(Mapping[str, object], latest[seed]["heldout"])["win_share"])
            for seed, row in diverse.items()
            if seed in latest
        ]
    )
    gates: dict[str, object] = {}
    if differences:
        latest_rows = grouped.get(latest_name, [])
        temporal_rows = grouped.get("balanced_temporal", [])
        selected_rows = grouped.get(diverse_name, [])

        def mean_row_metric(source: Sequence[Mapping[str, object]], key: str) -> float:
            values = [float(row.get(key, 0.0) or 0.0) for row in source]
            return sum(values) / len(values) if values else 0.0

        selected_js = mean_row_metric(
            selected_rows, "active_mean_pairwise_js_divergence"
        )
        baseline_js = max(
            mean_row_metric(latest_rows, "active_mean_pairwise_js_divergence"),
            mean_row_metric(temporal_rows, "active_mean_pairwise_js_divergence"),
        )
        selected_disagreement = mean_row_metric(
            selected_rows, "active_mean_pairwise_action_disagreement"
        )
        baseline_disagreement = max(
            mean_row_metric(latest_rows, "active_mean_pairwise_action_disagreement"),
            mean_row_metric(temporal_rows, "active_mean_pairwise_action_disagreement"),
        )
        js_ratio = (
            selected_js / baseline_js
            if baseline_js
            else (1_000_000.0 if selected_js > 0.0 else 0.0)
        )
        behavior_passed = js_ratio >= float(
            gate_values["min_behavioral_js_ratio"]
        ) and selected_disagreement - baseline_disagreement >= float(
            gate_values["min_action_disagreement_delta"]
        )
        matchup_differences: dict[str, list[float]] = {}
        for comparison in paired:
            per_game = comparison.get("per_game", [])
            if not isinstance(per_game, list):
                continue
            for game in per_game:
                game_row = _mapping(game, "paired game")
                matchup = str(game_row["matchup"])
                matchup_differences.setdefault(matchup, []).append(
                    float(game_row["difference"])
                )
        matchup_means = {
            matchup: sum(values) / len(values)
            for matchup, values in matchup_differences.items()
            if values
        }
        matchup_support = sum(value > 0.0 for value in matchup_means.values())
        gates["diversity_benefit"] = {
            "paired_seed_differences": differences,
            "paired_game_comparisons": list(paired_by_seed.values()),
            "mean_difference": sum(differences) / len(differences),
            "95_ci": _normal_interval(differences),
            "active_js_mean": selected_js,
            "reference_js_mean": baseline_js,
            "active_js_ratio": js_ratio,
            "active_action_disagreement_mean": selected_disagreement,
            "reference_action_disagreement_mean": baseline_disagreement,
            "matchup_mean_differences": matchup_means,
            "positive_matchup_count": matchup_support,
            "behavior_passed": behavior_passed,
            "passed": sum(differences) / len(differences)
            >= float(gate_values["min_heldout_difference"])
            and _normal_interval(differences)[0] > 0.0
            and min(differences) >= float(gate_values["max_seed_degradation"])
            and behavior_passed
            and matchup_support >= min(2, len(matchup_means)),
        }
    selected_name = _selected_condition_name(root, rows)
    selected = grouped.get(selected_name, [])
    gates["stable_adaptation"] = {
        "selected_condition": selected_name,
        "passed": bool(selected)
        and all(
            float(cast(Mapping[str, object], row["baseline"])["win_share"])
            >= float(gate_values["min_baseline_win_share"])
            and float(cast(Mapping[str, object], row["baseline"])["seat_spread"])
            <= float(gate_values["max_seat_spread"])
            and float(row.get("final_rating") or 0.0)
            >= float(gate_values["min_final_rating"])
            and float(row.get("final_minus_warmup_elo") or 0.0)
            >= float(gate_values["min_final_minus_warmup_elo"])
            and _direct_delta(row) > 0.0
            for row in selected
        ),
        "per_seed": [
            {
                "seed": row["seed"],
                "baseline_win_share": cast(Mapping[str, object], row["baseline"])[
                    "win_share"
                ],
                "seat_spread": cast(Mapping[str, object], row["baseline"])[
                    "seat_spread"
                ],
                "final_rating": row.get("final_rating"),
                "final_minus_warmup_elo": row.get("final_minus_warmup_elo"),
            }
            for row in selected
        ],
    }
    return {"aggregates": aggregates, "gates": gates}


def _selected_condition_name(
    root: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> str:
    """Select the fallback only when balanced-basic seat robustness requires it."""
    response_name = "balanced_response_diverse"
    response_rows = [row for row in rows if row.get("condition") == response_name]
    if response_rows and all(
        float(cast(Mapping[str, object], row["baseline"])["seat_spread"]) <= 0.05
        and float(cast(Mapping[str, object], row["baseline"])["win_share"]) >= 0.604
        for row in response_rows
    ):
        return response_name
    fallback = _mapping(root["fallback"], "fallback")
    fallback_name = str(fallback["name"])
    if any(row.get("condition") == fallback_name for row in rows):
        return fallback_name
    return response_name


def _direct_delta(row: Mapping[str, object]) -> float:
    direct = row.get("direct_final_warmup")
    if not isinstance(direct, dict):
        return float("-inf")
    return float(_mapping(direct, "direct comparison").get("delta", float("-inf")))


def _fallback_is_required(
    root: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> bool:
    fallback = _mapping(root["fallback"], "fallback")
    if fallback.get("enabled") is not True:
        return False
    response_rows = [
        row for row in rows if row.get("condition") == "balanced_response_diverse"
    ]
    gates = _mapping(root["gates"], "gates")
    return not response_rows or any(
        float(cast(Mapping[str, object], row["baseline"])["seat_spread"])
        > float(gates["max_seat_spread"])
        or float(cast(Mapping[str, object], row["baseline"])["win_share"])
        < float(gates["min_baseline_win_share"])
        for row in response_rows
    )


def _stage_config(root: Mapping[str, object], stage: str) -> Mapping[str, object]:
    return _mapping(root[stage], stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase7-follow-up-stability.yaml")
    )
    parser.add_argument(
        "--stage",
        choices=("screening", "confirmation", "mappo"),
        default="screening",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/phase7-follow-up-stability"),
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = load_config(args.config)
    stage = _stage_config(root, args.stage)
    seeds = _ints(stage["seeds"], f"{args.stage}.seeds")
    updates = int(stage["updates"])
    games = int(stage["games_per_seat"])
    focused_games = int(
        stage.get(
            "focused_tournament_games_per_lineup",
            stage.get("tournament_games_per_lineup", 1),
        )
    )
    full_games = int(stage.get("full_population_games_per_lineup", 1))
    workers = int(
        args.workers
        if args.workers is not None
        else _mapping(root["tournament"], "tournament")["workers"]
    )
    if workers < 1 or focused_games < 1 or full_games < 1:
        raise ValueError("workers and tournament game counts must be positive")
    if args.smoke:
        seeds = seeds[:1]
        updates = min(updates, 2)
        games = 1
        focused_games = 1
        full_games = 1
        workers = 1
    if args.stage == "mappo":
        confirmation_path = args.output_root / "confirmation-summary.json"
        if not args.smoke and not confirmation_path.is_file():
            raise ValueError(
                "MAPPO requires a completed stability confirmation summary"
            )
        if not args.smoke:
            confirmation = _mapping(
                json.loads(confirmation_path.read_text(encoding="utf-8")),
                "confirmation summary",
            )
            gates = _mapping(confirmation.get("gates", {}), "confirmation gates")
            stable = _mapping(gates.get("stable_adaptation", {}), "stable gate")
            if stable.get("passed") is not True:
                _write_json(
                    args.output_root / "mappo-decision.json",
                    {
                        "decision": "deferred",
                        "reason": (
                            "the confirmed league candidate did not pass "
                            "seat-robust stable adaptation gates"
                        ),
                    },
                )
                return
        mappo_stage = _stage_config(root, "mappo")
        mappo_rows = _run_mappo(
            root,
            args.output_root,
            seeds=_ints(mappo_stage["seeds"], "mappo.seeds")[: len(seeds)],
            updates=updates,
            games=games,
            focused_games=focused_games,
            workers=workers,
        )
        _write_json(
            args.output_root / "mappo-summary.json",
            {
                "experiment": str(root["experiment"]),
                "stage": "mappo",
                "results": mappo_rows,
            },
        )
        _write_json(
            args.output_root / "mappo-decision.json",
            {"decision": "evaluated", "results": mappo_rows},
        )
        return
    conditions = [
        _mapping(item, f"{args.stage}.condition[{index}]")
        for index, item in enumerate(
            _list(stage["conditions"], f"{args.stage}.conditions")
        )
    ]
    rows: list[dict[str, object]] = []
    for condition in conditions:
        for seed in seeds:
            run_dir = (
                args.output_root / args.stage / str(condition["name"]) / f"seed-{seed}"
            )
            manifest_path = run_dir / "manifest.json"
            if args.resume and manifest_path.is_file():
                manifest = _mapping(
                    json.loads(manifest_path.read_text(encoding="utf-8")),
                    "resume manifest",
                )
                if manifest.get("fixed_final_update") == updates:
                    validate_followup_manifest(manifest)
                    evaluation = _mapping(
                        json.loads(
                            (run_dir / "evaluation.json").read_text(encoding="utf-8")
                        ),
                        "resume evaluation",
                    )
                    diversity = _mapping(
                        json.loads(
                            (run_dir / "diversity.json").read_text(encoding="utf-8")
                        ),
                        "resume diversity",
                    )
                    rows.append(
                        {
                            "condition": condition["name"],
                            "seed": seed,
                            "baseline": _mapping(evaluation["baseline"], "baseline")[
                                "summary"
                            ],
                            "heldout": _mapping(evaluation["heldout"], "heldout")[
                                "summary"
                            ],
                            "final_rating": diversity.get("final_rating"),
                            "warmup_rating": diversity.get("warmup_rating"),
                            "final_minus_warmup_elo": diversity.get(
                                "final_minus_warmup_elo"
                            ),
                            "active_mean_pairwise_js_divergence": _active_metric(
                                diversity, "mean_pairwise_js_divergence"
                            ),
                            "active_mean_pairwise_action_disagreement": _active_metric(
                                diversity, "mean_pairwise_action_disagreement"
                            ),
                            "observation": manifest.get(
                                "observation",
                                _mapping(root["env"], "env")["observation"],
                            ),
                        }
                    )
                    continue
            observation_override = (
                str(condition["observation"]) if "observation" in condition else None
            )
            rows.append(
                _run_condition(
                    root,
                    condition,
                    seed,
                    args.output_root / args.stage,
                    updates=updates,
                    games=games,
                    focused_games=focused_games,
                    full_games=full_games,
                    workers=workers,
                    observation_override=observation_override,
                )
            )
    if _fallback_is_required(root, rows):
        fallback_values = _mapping(root["fallback"], "fallback")
        fallback_condition: dict[str, object] = {
            "name": str(fallback_values["name"]),
            "trainer": "league",
            "observation": str(fallback_values["observation"]),
            "population": {"retention_strategy": "novelty"},
        }
        fallback_rows: list[dict[str, object]] = []
        for seed in seeds:
            fallback_dir = (
                args.output_root
                / args.stage
                / str(fallback_condition["name"])
                / f"seed-{seed}"
            )
            if args.resume and (fallback_dir / "manifest.json").is_file():
                manifest = _mapping(
                    json.loads(
                        (fallback_dir / "manifest.json").read_text(encoding="utf-8")
                    ),
                    "fallback manifest",
                )
                if manifest.get("fixed_final_update") == updates:
                    validate_followup_manifest(manifest)
                    evaluation = _mapping(
                        json.loads(
                            (fallback_dir / "evaluation.json").read_text(
                                encoding="utf-8"
                            )
                        ),
                        "fallback evaluation",
                    )
                    diversity = _mapping(
                        json.loads(
                            (fallback_dir / "diversity.json").read_text(
                                encoding="utf-8"
                            )
                        ),
                        "fallback diversity",
                    )
                    fallback_rows.append(
                        {
                            "condition": fallback_condition["name"],
                            "seed": seed,
                            "baseline": _mapping(evaluation["baseline"], "baseline")[
                                "summary"
                            ],
                            "heldout": _mapping(evaluation["heldout"], "heldout")[
                                "summary"
                            ],
                            "final_rating": diversity.get("final_rating"),
                            "warmup_rating": diversity.get("warmup_rating"),
                            "final_minus_warmup_elo": diversity.get(
                                "final_minus_warmup_elo"
                            ),
                            "active_mean_pairwise_js_divergence": _active_metric(
                                diversity, "mean_pairwise_js_divergence"
                            ),
                            "active_mean_pairwise_action_disagreement": _active_metric(
                                diversity, "mean_pairwise_action_disagreement"
                            ),
                            "observation": fallback_condition["observation"],
                        }
                    )
                    continue
            fallback_rows.append(
                _run_condition(
                    root,
                    fallback_condition,
                    seed,
                    args.output_root / args.stage,
                    updates=updates,
                    games=games,
                    focused_games=focused_games,
                    full_games=full_games,
                    workers=workers,
                    observation_override=str(fallback_values["observation"]),
                )
            )
        rows.extend(fallback_rows)
        _write_json(
            args.output_root / args.stage / "fallback-decision.json",
            {
                "required": True,
                "reason": (
                    "balanced response-diverse screening failed the baseline "
                    "strength or seat-spread gate"
                ),
                "condition": fallback_condition["name"],
            },
        )
    else:
        _write_json(
            args.output_root / args.stage / "fallback-decision.json",
            {"required": False, "condition": None},
        )
    paired = _paired_heldout_comparisons(
        root, args.output_root / args.stage, rows, games=games
    )
    aggregate = _aggregate(root, rows, paired)
    _write_json(
        args.output_root / f"{args.stage}-summary.json",
        {
            "experiment": str(root["experiment"]),
            "stage": args.stage,
            "seeds": list(seeds),
            "updates": updates,
            "results": rows,
            "paired_heldout": paired,
            **aggregate,
        },
    )
    _write_json(
        args.output_root / "summary.json",
        {
            "experiment": str(root["experiment"]),
            "stage": args.stage,
            "results": rows,
            "paired_heldout": paired,
            **aggregate,
        },
    )


if __name__ == "__main__":
    main()
