"""Run the Phase 7 follow-up screening, confirmation, or MAPPO pilot."""

from __future__ import annotations

import argparse
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
    matchup_win_share_matrix,
    non_transitive_cycles,
    run_paired_round_robin,
    schedule_as_dict,
    validate_followup_manifest,
)
from flip7.evaluation.phase7 import TournamentParticipant
from flip7.experiment import (
    GateConfig,
    as_ints,
    as_list,
    as_mapping,
    as_pairs,
    as_strings,
    build_participants,
    cached_policy,
    mappo_config,
    run_standard_evaluations,
    sha256_file,
    training_config,
    write_json,
    write_stage_summary,
)
from flip7.training import (
    DiverseLeaguePPOTrainer,
    FollowUpLeagueConfig,
    MAPPOAgent,
    MAPPOTrainer,
    baseline_factories,
    write_followup_population,
    write_history,
    write_mappo_history,
)
from flip7.training.ppo import PPOConfig, PPOTrainer

_mapping = as_mapping
_list = as_list
_ints = as_ints
_strings = as_strings
_pairs = as_pairs
_write_json = write_json
_write_stage_summary = write_stage_summary
_sha256 = sha256_file
_training_config = training_config
_mappo_config = mappo_config
_cached_policy = cached_policy


def _condition_config(
    root: Mapping[str, object],
    condition: Mapping[str, object],
    seed: int,
    *,
    updates: int | None = None,
) -> tuple[PPOConfig, FollowUpLeagueConfig]:
    config = _training_config(root, seed, updates=updates)
    population = dict(_mapping(condition["population"], "condition.population"))
    opponents = _mapping(root["opponents"], "opponents")
    population["baseline_names"] = _strings(opponents["training"], "opponents.training")
    return config, FollowUpLeagueConfig(**population)


def _rating_from_elo(elo: Mapping[str, object], name: str) -> float | None:
    for value in _list(elo["ratings"], "tournament ratings"):
        row = _mapping(value, "tournament rating")
        if row.get("name") == name:
            return float(row["rating"])
    return None


def _aggregate_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    gate_config: GateConfig | None = None,
) -> dict[str, object]:
    """Aggregate trainable rows and evaluate predeclared follow-up gates."""
    gates_cfg = gate_config or GateConfig()
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        if isinstance(row.get("baseline"), dict) and isinstance(
            row.get("heldout"), dict
        ):
            grouped.setdefault(str(row["condition"]), []).append(row)

    aggregates: dict[str, object] = {}
    for condition, condition_rows in sorted(grouped.items()):
        baseline_rows = [
            cast(Mapping[str, object], row["baseline"]) for row in condition_rows
        ]
        heldout_rows = [
            cast(Mapping[str, object], row["heldout"]) for row in condition_rows
        ]

        def mean_metric(values: Sequence[Mapping[str, object]], key: str) -> float:
            return sum(float(value[key]) for value in values) / len(values)

        aggregates[condition] = {
            "seeds": [int(row["seed"]) for row in condition_rows],
            "baseline_win_share_mean": mean_metric(baseline_rows, "win_share"),
            "baseline_seat_spread_mean": mean_metric(baseline_rows, "seat_spread"),
            "heldout_win_share_mean": mean_metric(heldout_rows, "win_share"),
            "heldout_seat_spread_mean": mean_metric(heldout_rows, "seat_spread"),
            "mean_pairwise_js_divergence": (
                sum(float(row["mean_pairwise_js_divergence"]) for row in condition_rows)
                / len(condition_rows)
                if all("mean_pairwise_js_divergence" in row for row in condition_rows)
                else None
            ),
            "final_minus_warmup_elo": [
                row.get("final_minus_warmup_elo") for row in condition_rows
            ],
        }

    gates: dict[str, object] = {}
    latest = grouped.get("latest_only_dense", [])
    novelty = grouped.get("novelty_dense", [])
    if latest and novelty:
        latest_by_seed = {
            int(row["seed"]): cast(Mapping[str, object], row["heldout"])
            for row in latest
        }
        differences = [
            float(cast(Mapping[str, object], row["heldout"])["win_share"])
            - float(latest_by_seed[int(row["seed"])]["win_share"])
            for row in novelty
            if int(row["seed"]) in latest_by_seed
        ]
        if differences:
            mean_difference = sum(differences) / len(differences)
            standard_error = math.sqrt(
                sum((difference - mean_difference) ** 2 for difference in differences)
                / max(1, len(differences) - 1)
            ) / math.sqrt(len(differences))
            gates["novelty_vs_latest_heldout"] = {
                "paired_differences": differences,
                "mean_difference": mean_difference,
                "approximate_95_ci": [
                    mean_difference - 1.96 * standard_error,
                    mean_difference + 1.96 * standard_error,
                ],
                "passed": (
                    mean_difference >= gates_cfg.min_heldout_difference
                    and mean_difference - 1.96 * standard_error > 0
                    and min(differences) >= gates_cfg.max_seed_degradation
                ),
            }
    selected = "novelty_dense" if "novelty_dense" in grouped else None
    if selected is not None:
        selected_rows = grouped[selected]
        final_deltas = [
            float(row["final_minus_warmup_elo"])
            for row in selected_rows
            if row.get("final_minus_warmup_elo") is not None
        ]
        gates["stable_tournament_adaptation"] = {
            "passed": bool(final_deltas)
            and all(
                float(row["baseline"]["win_share"]) >= gates_cfg.min_baseline_win_share
                and float(row["baseline"]["seat_spread"]) <= gates_cfg.max_seat_spread
                and float(row.get("final_rating", 0.0)) >= gates_cfg.min_final_rating
                and float(row.get("final_minus_warmup_elo", 0.0))
                >= gates_cfg.min_final_minus_warmup_elo
                for row in selected_rows
            ),
            "all_seed_final_minus_warmup_elo": final_deltas,
            "all_seed_baseline_win_share": [
                float(row["baseline"]["win_share"]) for row in selected_rows
            ],
            "all_seed_seat_spread": [
                float(row["baseline"]["seat_spread"]) for row in selected_rows
            ],
        }
    return {"aggregates": aggregates, "gates": gates}


def _run_evaluation(
    root: Mapping[str, object],
    checkpoint: Path,
    games: int,
    seed_offset: int,
    *,
    updates: int,
    agent_factory: Any | None = None,
    workers: int = 1,
) -> tuple[dict[str, object], dict[str, object]]:
    observation = ObservationFamily(str(_mapping(root["env"], "env")["observation"]))
    return run_standard_evaluations(
        root,
        checkpoint,
        games=games,
        seed_offset=seed_offset,
        observation=observation,
        policy_factory=agent_factory,
        workers=workers,
    )


def _run_training_condition(
    root: Mapping[str, object],
    condition: Mapping[str, object],
    seed: int,
    output_root: Path,
    *,
    updates: int,
    games: int,
    tournament_games: int,
    tournament_workers: int,
    workers: int = 1,
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
    config, league_config = _condition_config(root, condition, seed, updates=updates)
    trainer: DiverseLeaguePPOTrainer | PPOTrainer
    trainer_name = str(condition.get("trainer", "league"))
    if trainer_name == "ppo":
        trainer = PPOTrainer(config, opponent_names=league_config.baseline_names)
    elif trainer_name == "league":
        trainer = DiverseLeaguePPOTrainer(config, league_config=league_config)
    else:
        raise ValueError(f"unsupported follow-up trainer: {trainer_name}")
    history = trainer.train(checkpoint)
    write_history(history_path, history)
    snapshots: Sequence[Any] = ()
    archived: Sequence[Any] = ()
    state_bank = None
    warmup = None
    if isinstance(trainer, DiverseLeaguePPOTrainer):
        write_followup_population(population_path, trainer.league)
        snapshots = trainer.league.snapshots
        archived = trainer.league.archived_snapshots
        state_bank = trainer.league.state_bank
        warmup = trainer.league.warmup_anchor
    else:
        _write_json(
            population_path,
            {
                "algorithm": "ppo",
                "snapshots": [],
                "warmup_anchor": None,
                "opponent_exposure": {},
            },
        )
    baseline, heldout = _run_evaluation(
        root, checkpoint, games, seed * 10_000, updates=updates, workers=workers
    )
    observation = ObservationFamily(config.observation)
    participants = build_participants(
        checkpoint, observation, snapshots, warmup, include_snapshots=True
    )
    checkpoint_paths = {"final": checkpoint}
    checkpoint_paths.update(
        {snapshot.policy_id: snapshot.path for snapshot in snapshots}
    )
    if warmup is not None:
        checkpoint_paths[warmup.policy_id] = warmup.path
    tournament_values = _mapping(root["tournament"], "tournament")
    schedule = build_paired_schedule(
        tuple(participant.name for participant in participants),
        games=tournament_games,
        seed=int(tournament_values["seed"]) + seed,
    )
    tournament, schedule = run_paired_round_robin(
        participants,
        games=tournament_games,
        seed=int(tournament_values["seed"]) + seed,
        initial_rating=float(tournament_values["initial_rating"]),
        k_factor=float(tournament_values["k_factor"]),
        schedule=schedule,
        workers=tournament_workers,
        checkpoint_paths=checkpoint_paths,
    )
    direct = None
    if warmup is not None:
        final_participant = next(
            participant for participant in participants if participant.name == "final"
        )
        direct = direct_final_warmup_comparison(
            final_participant,
            TournamentParticipant(
                warmup.policy_id,
                lambda path=warmup.path: PPOAgent.from_checkpoint(
                    path, deterministic=True
                ),
                warmup.observation,
            ),
            tuple(
                participant
                for participant in participants
                if participant.name in {"random", "threshold", "risk", "ev", "dp"}
            ),
            games=tournament_games,
            seed=int(tournament_values["seed"]) + seed + 900_000,
        )
    diversity = (
        analyze_snapshots(archived, state_bank)
        if state_bank is not None
        else {
            "policies": [],
            "mean_pairwise_js_divergence": 0.0,
            "mean_pairwise_action_disagreement": 0.0,
        }
    )
    diversity["archived_snapshots"] = [snapshot.as_dict() for snapshot in archived]
    diversity["active_snapshots"] = [snapshot.as_dict() for snapshot in snapshots]
    diversity["state_bank"] = state_bank.as_dict() if state_bank is not None else None
    diversity["snapshot_elo"] = tournament.elo
    final_rating = _rating_from_elo(tournament.elo, "final")
    warmup_rating = (
        _rating_from_elo(tournament.elo, warmup.policy_id)
        if warmup is not None
        else None
    )
    diversity["final_rating"] = final_rating
    diversity["warmup_rating"] = warmup_rating
    diversity["final_minus_warmup_elo"] = (
        final_rating - warmup_rating
        if final_rating is not None and warmup_rating is not None
        else None
    )
    diversity["snapshot_elo_progression"] = [
        {
            "policy_id": snapshot.policy_id,
            "update": snapshot.update,
            "final_tournament_rating": _rating_from_elo(
                tournament.elo, snapshot.policy_id
            ),
        }
        for snapshot in archived
    ]
    diversity["matchup_win_share_matrix"] = matchup_win_share_matrix(tournament)
    diversity["non_transitive_cycles"] = non_transitive_cycles(
        cast(Mapping[str, Mapping[str, float]], diversity["matchup_win_share_matrix"])
    )
    diversity["tournament_schedule"] = {
        "games_per_lineup": tournament_games,
        "games": len(schedule),
        "seed": int(tournament_values["seed"]) + seed,
        "workers": tournament_workers,
    }
    _write_json(diversity_path, diversity)
    tournament_payload = tournament.as_dict() | {
        "schedule": schedule_as_dict(schedule),
        "direct_final_warmup": direct,
        "final_rating": final_rating,
        "warmup_rating": warmup_rating,
        "final_minus_warmup_elo": diversity["final_minus_warmup_elo"],
    }
    _write_json(tournament_path, tournament_payload)
    evaluation_payload: dict[str, object] = {
        "experiment": str(root["experiment"]),
        "condition": name,
        "seed": seed,
        "updates": updates,
        "baseline": baseline,
        "heldout": heldout,
    }
    _write_json(evaluation_path, evaluation_payload)
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
        "state_bank": state_bank.as_dict() if state_bank is not None else None,
        "schedule_games": len(schedule),
        "tournament_workers": tournament_workers,
        "fixed_final_update": updates,
    }
    _write_json(manifest_path, manifest)
    validate_followup_manifest(manifest)
    return {
        "condition": name,
        "seed": seed,
        "baseline": baseline["summary"],
        "heldout": heldout["summary"],
        "final_update": updates,
        "archived_snapshots": len(archived),
        "active_snapshots": len(snapshots),
        "mean_pairwise_js_divergence": diversity["mean_pairwise_js_divergence"],
        "tournament": tournament.elo,
        "direct_final_warmup": direct,
        "final_rating": final_rating,
        "warmup_rating": warmup_rating,
        "final_minus_warmup_elo": diversity["final_minus_warmup_elo"],
    }


def _load_completed_condition(
    run_dir: Path, *, condition: str, seed: int, updates: int
) -> dict[str, object] | None:
    """Load a complete fixed-update run for safe interruption recovery."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = _mapping(manifest_value, "resume manifest")
        validate_followup_manifest(manifest)
        if (
            manifest.get("condition") != condition
            or manifest.get("seed") != seed
            or manifest.get("fixed_final_update") != updates
        ):
            return None
        evaluation = _mapping(
            json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8")),
            "resume evaluation",
        )
        baseline = _mapping(evaluation["baseline"], "resume baseline")
        heldout = _mapping(evaluation["heldout"], "resume heldout")
        diversity = _mapping(
            json.loads((run_dir / "diversity.json").read_text(encoding="utf-8")),
            "resume diversity",
        )
        population = _mapping(
            json.loads((run_dir / "population.json").read_text(encoding="utf-8")),
            "resume population",
        )
        tournament = _mapping(
            json.loads((run_dir / "tournament.json").read_text(encoding="utf-8")),
            "resume tournament",
        )
        archived = population.get("archived_snapshots", [])
        active = population.get("snapshots", [])
        if not isinstance(archived, list) or not isinstance(active, list):
            return None
        return {
            "condition": condition,
            "seed": seed,
            "baseline": baseline["summary"],
            "heldout": heldout["summary"],
            "final_update": updates,
            "archived_snapshots": len(archived),
            "active_snapshots": len(active),
            "mean_pairwise_js_divergence": diversity.get(
                "mean_pairwise_js_divergence", 0.0
            ),
            "tournament": tournament.get("elo", {}),
            "direct_final_warmup": tournament.get("direct_final_warmup"),
            "final_rating": diversity.get("final_rating"),
            "warmup_rating": diversity.get("warmup_rating"),
            "final_minus_warmup_elo": diversity.get("final_minus_warmup_elo"),
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _run_mappo(
    root: Mapping[str, object],
    output_root: Path,
    *,
    updates: int,
    games: int,
    seeds: Sequence[int],
    workers: int = 1,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        run_dir = output_root / "mappo" / f"seed-{seed}"
        checkpoint = run_dir / "checkpoint.pt"
        history_path = run_dir / "training.json"
        population_path = run_dir / "population.json"
        diversity_path = run_dir / "diversity.json"
        evaluation_path = run_dir / "evaluation.json"
        tournament_path = run_dir / "tournament.json"
        manifest_path = run_dir / "manifest.json"
        trainer = MAPPOTrainer(_mappo_config(root, seed, updates=updates))
        history = trainer.train(checkpoint)
        write_mappo_history(history_path, history)
        evaluation, heldout = _run_evaluation(
            root,
            checkpoint,
            games,
            seed * 10_000,
            updates=updates,
            agent_factory=lambda path: MAPPOAgent.from_checkpoint(
                path, deterministic=True
            ),
            workers=workers,
        )
        observation = ObservationFamily(
            str(_mapping(root["env"], "env")["observation"])
        )
        participants = [
            TournamentParticipant(name, factory, ObservationFamily.DECK_AWARE)
            for name, factory in baseline_factories().items()
        ]
        participants.append(
            TournamentParticipant(
                "mappo",
                lambda path=checkpoint: MAPPOAgent.from_checkpoint(
                    path, deterministic=True
                ),
                observation,
            )
        )
        tournament_values = _mapping(root["tournament"], "tournament")
        tournament, schedule = run_paired_round_robin(
            tuple(participants),
            games=int(tournament_values["games_per_lineup"]),
            seed=int(tournament_values["seed"]) + seed,
            initial_rating=float(tournament_values["initial_rating"]),
            k_factor=float(tournament_values["k_factor"]),
        )
        _write_json(
            tournament_path,
            tournament.as_dict() | {"schedule": schedule_as_dict(schedule)},
        )
        _write_json(
            population_path,
            {
                "algorithm": "mappo",
                "trainable_seats": int(root["players"]),
                "snapshots": [],
                "warmup_anchor": None,
            },
        )
        _write_json(
            diversity_path,
            {
                "algorithm": "mappo",
                "state_bank": None,
                "policies": [],
                "matchup_win_share_matrix": {},
                "non_transitive_cycles": [],
            },
        )
        _write_json(evaluation_path, {"baseline": evaluation, "heldout": heldout})
        manifest = {
            "experiment": str(root["experiment"]),
            "condition": "mappo",
            "seed": seed,
            "checkpoint": str(checkpoint),
            "training_history": str(history_path),
            "population": str(population_path),
            "diversity": str(diversity_path),
            "evaluation": str(evaluation_path),
            "tournament": str(tournament_path),
            "manifest": str(manifest_path),
            "critic_input_size": trainer.critic_input_size,
            "fixed_final_update": updates,
        }
        _write_json(manifest_path, manifest)
        validate_followup_manifest(manifest)
        rows.append(
            {
                "seed": seed,
                "baseline": evaluation["summary"],
                "heldout": heldout["summary"],
                "critic_input_size": trainer.critic_input_size,
            }
        )
    return rows


def _reference_rows(
    root: Mapping[str, object], names: Sequence[str], seeds: Sequence[int]
) -> list[dict[str, object]]:
    references = _mapping(root["references"], "references")
    rows: list[dict[str, object]] = []
    for name in names:
        reference_dir = Path(str(references[name]))
        for seed in seeds:
            evaluation_path = reference_dir / f"seed-{seed}" / "evaluation.json"
            payload = _mapping(
                json.loads(evaluation_path.read_text(encoding="utf-8")),
                "reference evaluation",
            )
            summary = _mapping(payload.get("summary", {}), "reference summary")
            rows.append(
                {
                    "condition": name,
                    "seed": seed,
                    "reference": str(evaluation_path),
                    "baseline": summary.get("baseline_rotated", {}),
                    "heldout": summary.get("heldout_rotated", {}),
                    "immutable": True,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase7-follow-up.yaml")
    )
    parser.add_argument(
        "--stage", choices=("screening", "confirmation", "mappo"), default="screening"
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/phase7-follow-up")
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel CPU workers for evaluation matchups (default: 1)",
    )
    parser.add_argument(
        "--tournament-workers",
        type=int,
        default=None,
        help="parallel CPU workers for independent tournament games",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse complete fixed-update runs after an interrupted stage",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the configured stage with small deterministic budgets",
    )
    args = parser.parse_args()
    root = load_config(args.config)
    stage = _mapping(root[args.stage], args.stage)
    seeds = _ints(stage["seeds"], f"{args.stage}.seeds")
    updates = int(stage["updates"])
    games = int(stage["games_per_seat"])
    tournament_games = int(stage["tournament_games_per_lineup"])
    tournament_values = _mapping(root["tournament"], "tournament")
    configured_workers = int(tournament_values.get("workers", 1))
    tournament_workers = (
        int(args.tournament_workers)
        if args.tournament_workers is not None
        else configured_workers
    )
    if tournament_workers < 1:
        raise ValueError("tournament workers must be positive")
    if args.smoke:
        seeds = seeds[:1]
        updates = min(updates, 12 if args.stage == "screening" else 2)
        games = min(games, 1)
        tournament_games = 1
        if args.tournament_workers is None:
            tournament_workers = 1
    if (
        args.stage == "mappo"
        and not args.smoke
        and not (args.output_root / "confirmation-summary.json").is_file()
    ):
        raise ValueError("MAPPO requires a completed follow-up confirmation summary")
    if args.stage == "mappo":
        rows = _run_mappo(
            root,
            args.output_root,
            updates=updates,
            games=games,
            seeds=seeds,
            workers=args.workers,
        )
        _write_stage_summary(
            args.output_root,
            args.stage,
            {
                "experiment": str(root["experiment"]),
                "stage": args.stage,
                "results": rows,
            },
        )
        return
    conditions = _list(stage["conditions"], f"{args.stage}.conditions")
    reference_names = tuple(
        str(_mapping(item, "condition")["name"])
        for item in conditions
        if str(_mapping(item, "condition").get("kind", "train")) == "reference"
    )
    rows = _reference_rows(root, reference_names, seeds) if reference_names else []
    for item in conditions:
        condition = _mapping(item, "condition")
        if str(condition.get("kind", "train")) == "reference":
            continue
        for seed in seeds:
            if args.resume:
                completed = _load_completed_condition(
                    args.output_root
                    / args.stage
                    / str(condition["name"])
                    / f"seed-{seed}",
                    condition=str(condition["name"]),
                    seed=seed,
                    updates=updates,
                )
                if completed is not None:
                    rows.append(completed)
                    continue
            rows.append(
                _run_training_condition(
                    root,
                    condition,
                    seed,
                    args.output_root / args.stage,
                    updates=updates,
                    games=games,
                    tournament_games=tournament_games,
                    tournament_workers=tournament_workers,
                    workers=args.workers,
                )
            )
    _write_stage_summary(
        args.output_root,
        args.stage,
        {
            "experiment": str(root["experiment"]),
            "stage": args.stage,
            "seeds": list(seeds),
            "updates": updates,
            "results": rows,
            **_aggregate_rows(rows, gate_config=GateConfig.from_config(root)),
        },
    )


if __name__ == "__main__":
    main()
